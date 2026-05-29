"""
utils/thread_store.py

Maintains a lightweight `sidebar_index` table in Postgres for
fast sidebar loads — no checkpoint blob parsing needed.

Table schema (auto-created on first run):

    CREATE TABLE sidebar_index (
        thread_id        TEXT PRIMARY KEY,
        user_id          TEXT NOT NULL,
        title            TEXT NOT NULL DEFAULT 'New conversation',
        created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_message_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_preview     TEXT NOT NULL DEFAULT ''
    );

    CREATE INDEX IF NOT EXISTS idx_sidebar_user
        ON sidebar_index(user_id, last_message_at DESC);

This index is what makes the sidebar query instant:
    SELECT * FROM sidebar_index WHERE user_id = $1
    ORDER BY last_message_at DESC

One indexed row per conversation — no blob deserialization,
no join with checkpoints, no full table scan.
"""

import os
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()


# ── Connection ─────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(
        os.getenv("POSTGRES_URL"),
        cursor_factory=psycopg2.extras.RealDictCursor   # rows as dicts
    )


# ── One-time setup (call on startup) ──────────────────────────

def setup_sidebar_table() -> None:
    """
    Create sidebar_index table and index if they don't exist.
    Safe to call on every startup — fully idempotent.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_chat_threads (
                    thread_id        TEXT PRIMARY KEY,
                    user_id          TEXT NOT NULL,
                    title            TEXT NOT NULL DEFAULT 'New conversation',
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    last_message_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    last_preview     TEXT NOT NULL DEFAULT ''
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_sidebar_user
                    ON user_chat_threads(user_id, last_message_at DESC);
            """)
            conn.commit()
    
    finally:
        conn.close()


# ── WRITE operations ───────────────────────────────────────────

def create_new_thread(thread_id: str, user_id: str) -> None:
    """
    Insert a new row when the user starts a new conversation.
    Title starts as 'New conversation' — updated after first message.
    Called from POST /sessions/new.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_chat_threads
                    (thread_id, user_id, title, created_at, last_message_at, last_preview)
                VALUES
                    (%s, %s, 'New conversation', now(), now(), '')
                ON CONFLICT (thread_id) DO NOTHING;
            """, (thread_id, user_id))
            conn.commit()
    finally:
        conn.close()


def update_thread_after_turn(
    thread_id: str,
    user_message: str,
    assistant_reply: str
) -> None:
    """
    Called at the end of every POST /chat turn.

    - If title is still 'New conversation', set it from the
      first user message (truncated to 60 chars) — exactly
      like ChatGPT titles the conversation.
    - Always update last_message_at and last_preview.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE user_chat_threads
                SET
                    title = CASE
                        WHEN title = 'New conversation'
                        THEN %s
                        ELSE title
                    END,
                    last_message_at = now(),
                    last_preview    = %s
                WHERE thread_id = %s;
            """, (
                _truncate(user_message, 60),
                _truncate(assistant_reply, 80),
                thread_id,
            ))
            conn.commit()
    finally:
        conn.close()


def delete_thread(thread_id: str) -> None:
    """
    Remove conversation from sidebar AND from LangGraph checkpoints.
    Called from DELETE /sessions/{thread_id}.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            # Remove sidebar entry
            cur.execute(
                "DELETE FROM user_chat_threads WHERE thread_id = %s",
                (thread_id,)
            )
            # Remove LangGraph checkpoint rows
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                cur.execute(
                    f"DELETE FROM {table} WHERE thread_id = %s",
                    (thread_id,)
                )
            conn.commit()
    finally:
        conn.close()


# ── READ operations ────────────────────────────────────────────

def get_user_threads(user_id: str) -> list[dict]:
    """
    Return all conversations for a user, newest first.
    This is the ONLY query that runs on every sidebar page load.
    It hits the indexed sidebar_index table — fast even at scale.

    Returns:
        [
            {
                "thread_id":        "abc123",
                "title":            "What is diabetes treatment?",
                "created_at":       "2026-05-26T10:00:00+00:00",
                "last_message_at":  "2026-05-26T10:45:00+00:00",
                "last_preview":     "Diabetes treatment involves..."
            },
            ...
        ]
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    thread_id,
                    title,
                    created_at,
                    last_message_at,
                    last_preview
                FROM user_chat_threads
                WHERE user_id = %s
                ORDER BY last_message_at DESC;
            """, (user_id,))
            rows = cur.fetchall()

        return [
            {
                "thread_id":       row["thread_id"],
                "title":           row["title"],
                "created_at":      row["created_at"].isoformat(),
                "last_message_at": row["last_message_at"].isoformat(),
                "last_preview":    row["last_preview"],
            }
            for row in rows
        ]
    finally:
        conn.close()


# ── Internal helpers ───────────────────────────────────────────

def _truncate(text: str, length: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    return text[:length] + "…" if len(text) > length else text