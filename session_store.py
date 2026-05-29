"""
utils/session_store.py

All Redis logic in one place. Two key types only:

  user:{user_id}:threads   → list of thread metadata (sidebar)
  session:{thread_id}      → per-conversation agent state

No TTL on either — sessions live until the user explicitly
starts a new chat or clears history. Image data lives inside
the session object, cleared after each chat turn that uses it.
"""

import json
import redis
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# ── Single shared Redis connection pool ────────────────────────
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", None),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
)


# ══════════════════════════════════════════════════════════════
#  KEY HELPERS
# ══════════════════════════════════════════════════════════════

def _thread_list_key(user_id: str) -> str:
    """Key that holds the sidebar thread list for a user."""
    return f"user:{user_id}:threads"

def _session_key(thread_id: str) -> str:
    """Key that holds agent state for one conversation."""
    return f"session:{thread_id}"


# ══════════════════════════════════════════════════════════════
#  THREAD LIST  (sidebar)
# ══════════════════════════════════════════════════════════════

def get_user_threads(user_id: str) -> list[dict]:
    """
    Return all threads for a user, newest first.
    Used to render the sidebar on page load.

    Each item:
    {
        thread_id     : str,
        title         : str,   # first user message, truncated to 60 chars
        created_at    : str,   # ISO 8601
        last_message_at: str,  # ISO 8601
        last_preview  : str    # last assistant reply, truncated to 80 chars
    }
    """
    raw = redis_client.get(_thread_list_key(user_id))
    if not raw:
        return []
    threads = json.loads(raw)
    # Sort newest first by last_message_at
    return sorted(threads, key=lambda t: t["last_message_at"], reverse=True)


def add_thread_to_user(user_id: str, thread_id: str, first_message: str) -> None:
    """
    Add a new thread entry to the user's sidebar list.
    Called when POST /sessions/new is hit.
    """
    threads = get_user_threads(user_id)
    now = _now()
    threads.append({
        "thread_id": thread_id,
        "title": _truncate(first_message, 60) if first_message else "New conversation",
        "created_at": now,
        "last_message_at": now,
        "last_preview": "",
    })
    redis_client.set(_thread_list_key(user_id), json.dumps(threads))


def update_thread_preview(user_id: str, thread_id: str,
                          user_message: str, assistant_reply: str) -> None:
    """
    After each chat turn, update the sidebar entry:
    - Set title from first user message (if still default)
    - Update last_message_at timestamp
    - Update last_preview from assistant reply

    Called at the end of POST /chat.
    """
    threads = get_user_threads(user_id)
    for t in threads:
        if t["thread_id"] == thread_id:
            if t["title"] == "New conversation":
                t["title"] = _truncate(user_message, 60)
            t["last_message_at"] = _now()
            t["last_preview"] = _truncate(assistant_reply, 80)
            break
    redis_client.set(_thread_list_key(user_id), json.dumps(threads))


def delete_thread_from_user(user_id: str, thread_id: str) -> None:
    """Remove one thread from the user's sidebar list."""
    threads = get_user_threads(user_id)
    threads = [t for t in threads if t["thread_id"] != thread_id]
    redis_client.set(_thread_list_key(user_id), json.dumps(threads))


# ══════════════════════════════════════════════════════════════
#  SESSION  (agent state per conversation)
# ══════════════════════════════════════════════════════════════
def _default_session(user_id: str, thread_id: str) -> dict:
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "indexed_docs": [],
        "image": None
    }


def get_session(thread_id: str, user_id: str) -> dict:
    """
    Load session from Redis.
    If not found, return a blank default (does NOT write to Redis yet).
    Pass user_id only when creating a brand-new session.
    """
    raw = redis_client.get(thread_id)
    if raw:
        return json.loads(raw)
    
    return _default_session(user_id, thread_id)


def save_session(session: dict) -> None:
    """
    Persist session to Redis. No TTL — lives until explicitly deleted.
    Always call this after mutating the session dict.
    """
    session["last_active_at"] = _now()
    redis_client.set(_session_key(session["thread_id"]), json.dumps(session))


def delete_session(thread_id: str) -> None:
    """Delete session data. Called when user clears a conversation."""
    redis_client.delete(_session_key(thread_id))


def consume_image(session: dict) -> Optional[list]:
    """
    Return image_data and clear it from the session in one step.
    Call this at the start of each /chat turn — image is single-use
    PER TURN but persists across turns until explicitly cleared by
    the user uploading a new image or starting a new chat.

    Wait — actually images should persist for the whole conversation
    (user can ask multiple questions about the same image). So we
    return the data but do NOT delete it. The frontend decides when
    to clear by calling POST /sessions/{thread_id}/clear-image.
    """
    return session.get("image_data")


def clear_image(session: dict) -> dict:
    """
    Explicitly clear image from session.
    Called when user uploads a new image (replaces old) or
    starts a new chat.
    """
    session["uploaded_image"] = False
    session["image_data"] = None
    return session


def add_doc_to_session(session: dict, doc_entry: dict) -> dict:
    """
    Add an indexed doc to the session.
    doc_entry = { doc_id, topic, filename, summary }
    Also updates external_kb_meta to point to this latest doc.
    """
    # Avoid duplicates by filename
    session["indexed_docs"] = [
        d for d in session["indexed_docs"]
        if d["filename"] != doc_entry["filename"]
    ]
    session["indexed_docs"].append(doc_entry)

    # external_kb_meta always points to the most recently uploaded doc
    # The full list is in indexed_docs for the router to use
    session["external_kb_meta"] = {
        "available": True,
        "topic": doc_entry["topic"],
        "summary": doc_entry["summary"],
        "all_docs": session["indexed_docs"],
    }
    return session


# ══════════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════════

def ping() -> bool:
    try:
        redis_client.ping()
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
#  INTERNAL UTILS
# ══════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _truncate(text: str, length: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    return text[:length] + "…" if len(text) > length else text
