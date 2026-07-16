"""
FastAPI Service — Production Grade v3

Storage responsibilities:
  Postgres  →  sidebar_index table (fast sidebar loads)
              LangGraph checkpoints (full message history)
  Redis     →  session:{thread_id} (agent runtime state only)
  Disk      →  /tmp/img_{thread_id}.{ext} (image bytes)
"""

import os
import tempfile
import base64
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

from src.agent.langgraph_agent import ai_agent
from src.db_connections.pinecone.vectorstore import add_doc_to_vectorstore
from utils.common import generate_thread_id, load_pdf, clean_text, generate_summary
from utils.checkpointer import get_postgres_checkpointer
from utils.session_store import get_session, save_session, delete_session, ping
from utils.thread_store import (
    setup_sidebar_table,
    create_new_thread,
    update_thread_after_turn,
    delete_thread,
    get_user_threads,
)
from src.logger.logging import logging


# ══════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify Redis
    if not ping():
        raise RuntimeError("Redis unreachable. Check REDIS_* env vars.")
    logging.info("Redis: OK")

    # Create sidebar_index table if it doesn't exist
    setup_sidebar_table()
    logging.info("Postgres sidebar_index: OK")

    yield


app = FastAPI(title="Multi-Agent AI API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
#  SESSION ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/sessions/new", response_model=NewSessionResponse, tags=["Sessions"])
def create_new_session(user_id: str):
    """
    Start a new conversation.

    1. Generate thread_id
    2. Insert row into user_chat_threads (Postgres)
    3. Initialise blank agent state (Redis)

    Query param: ?user_id=<user_id>
    """
    thread_id = generate_thread_id()

    # Postgres — sidebar entry
    create_new_thread(thread_id, user_id)

    # Redis — blank agent state
    session = get_session(thread_id, user_id)
    save_session(session)

    logging.info(f"New session [{thread_id}] for user [{user_id}]")
    return NewSessionResponse(thread_id=thread_id)


@app.get("/{user_id}/threads", response_model=SidebarResponse)
def get_sidebar(user_id: str):
    """
    Load the sidebar on page login/load.
    Hits the indexed sidebar_index table — fast at any scale.
    Returns all conversations newest first with title + preview.
    """
    threads = get_user_threads(user_id)
    return SidebarResponse(threads=[ThreadItem(**t) for t in threads])


@app.get("/sessions/{thread_id}/messages", response_model=SessionMessagesResponse, tags=["Sessions"])
def get_messages(thread_id: str):
    """
    Load full message history when user clicks a conversation.
    Reads from LangGraph Postgres checkpointer.
    """
    try:
        state = ai_agent.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        raw = state.values.get("messages", [])
        messages = []
        for msg in raw:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else "[image]"
                messages.append(MessageItem(role="user", content=content))
            elif isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                messages.append(MessageItem(role="assistant", content=content))
        return SessionMessagesResponse(thread_id=thread_id, messages=messages)
    
    except Exception as e:
        logging.error(f"Error loading messages [{thread_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{thread_id}", tags=["Sessions"])
def delete_conversation(thread_id: str):
    """
    Delete a conversation completely.
    - Removes sidebar_index row (Postgres)
    - Removes LangGraph checkpoints (Postgres)
    - Removes agent state (Redis)
    - Removes image temp file (disk)
    """
    session = get_session(thread_id)

    # Clean up image file from disk if it exists
    _cleanup_image_file(session)

    delete_thread(thread_id)      # Postgres: sidebar + checkpoints
    delete_session(thread_id)     # Redis: agent state

    return {"status": "deleted", "thread_id": thread_id}


# ══════════════════════════════════════════════════════════════
#  CHAT
# ══════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest, x_user_id: str = Header(...)):
    """
    Send a message, get the agent's reply.

    Header required: X-User-Id: <user_id>

    Flow:
    1. Load agent state from Redis
    2. Load image from disk if uploaded
    3. Invoke LangGraph (auto-saves to Postgres checkpointer)
    4. Update sidebar_index row in Postgres (title + preview)
    """
    session = get_session(req.thread_id)

    # Load image bytes from disk — NOT from Redis
    image_data = _load_image_for_gemini(session)

    config = {
        "configurable": {"thread_id": req.thread_id},
        "metadata": {
            "thread_id": req.thread_id,
            "user_id": x_user_id,
        },
        "run_name": "chat_turn",
    }

    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "indexed_docs": session.get("indexed_docs", []),
        "image_data": image_data or [],
    }

    try:
        response = ai_agent.invoke(initial_state, config=config)
        reply = str(response["messages"][-1].content)

        # Update sidebar title + preview in Postgres
        update_thread_after_turn(
            thread_id=req.thread_id,
            user_message=req.message,
            assistant_reply=reply,
        )

        logging.info(f"[{req.thread_id}] Chat turn complete.")
        return ChatResponse(thread_id=req.thread_id, reply=reply)

    except Exception as e:
        logging.error(f"Chat error [{req.thread_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
#  FILE UPLOADS
# ══════════════════════════════════════════════════════════════

@app.post("/upload/pdf", response_model=UploadResponse, tags=["Uploads"])
async def upload_pdf(thread_id: str, file: UploadFile = File(...)):
    """
    Upload PDF → extract → summarise → index into Pinecone.
    Metadata stored in Redis session. No PDF bytes stored anywhere.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    session = get_session(thread_id)

    already = any(d["filename"] == file.filename for d in session["indexed_docs"])
    if already:
        return UploadResponse(
            thread_id=thread_id,
            status="skipped",
            detail=f"{file.filename} is already indexed in this session.",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        documents = load_pdf(tmp_path)
        if not documents:
            raise ValueError("PDF is empty or could not be parsed.")

        extracted_text = clean_text("\n".join(d.page_content for d in documents))
        summary = generate_summary(extracted_text)

        add_doc_to_vectorstore(index_name=summary["topic"], content=extracted_text)

        doc_entry = {
            "doc_id": generate_thread_id(),
            "topic": summary["topic"],
            "filename": file.filename,
            "summary": summary["summary"],
        }

        session["indexed_docs"].append(doc_entry)
        session["external_kb_meta"] = {
            "available": True,
            "topic": summary["topic"],
            "summary": summary["summary"],
            "all_docs": session["indexed_docs"],
        }
        save_session(session)

        logging.info(f"[{thread_id}] PDF indexed: {summary['topic']}")
        return UploadResponse(
            thread_id=thread_id,
            status="success",
            topic=summary["topic"],
            summary=summary["summary"],
        )

    except Exception as e:
        logging.error(f"PDF upload error [{thread_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/upload/image", response_model=UploadResponse, tags=["Uploads"])
async def upload_image(thread_id: str, file: UploadFile = File(...)):
    """
    Upload image for vision analysis.

    Image bytes saved to disk (/tmp/).
    Only the file path + mime_type stored in Redis session (~60 bytes).
    Image persists for the whole conversation until new chat or
    explicit clear.
    """
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in {"png", "jpg", "jpeg"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    try:
        bytes_data = await file.read()

        # Write to temp file — tiny footprint in Redis
        tmp_path = f"/tmp/img_{thread_id}.{ext}"
        with open(tmp_path, "wb") as f:
            f.write(bytes_data)

        # Store only path + mime_type in Redis
        session = get_session(thread_id)
        _cleanup_image_file(session)   # remove old image if replacing
        session["uploaded_image"] = True
        session["image_data"] = {
            "path": tmp_path,
            "mime_type": file.content_type or f"image/{ext}",
        }
        save_session(session)

        logging.info(f"[{thread_id}] Image saved to {tmp_path}")
        return UploadResponse(
            thread_id=thread_id,
            status="success",
            detail="Image ready. Ask your question via /chat.",
        )
    except Exception as e:
        logging.error(f"Image upload error [{thread_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{thread_id}/clear-image", tags=["Sessions"])
def clear_image(thread_id: str):
    """Explicitly clear the uploaded image from a session."""
    session = get_session(thread_id)
    _cleanup_image_file(session)
    session["uploaded_image"] = False
    session["image_data"] = None
    save_session(session)
    return {"status": "cleared", "thread_id": thread_id}


# ══════════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════════

@app.get("/health", tags=["Health"])
def health():
    return {
        "api": "ok",
        "redis": "ok" if ping() else "unreachable",
    }


# ══════════════════════════════════════════════════════════════
#  IMAGE HELPERS
# ══════════════════════════════════════════════════════════════

def _load_image_for_gemini(session: dict) -> Optional[list]:
    """
    Read image bytes from disk and return in the format
    Gemini expects: [{"mime_type": "image/jpeg", "data": bytes}]
    Returns None if no image uploaded.
    """
    image_ref = session.get("image_data")
    if not image_ref or not isinstance(image_ref, dict):
        return None

    path = image_ref.get("path")
    if not path or not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return [{"mime_type": image_ref["mime_type"], "data": f.read()}]


def _cleanup_image_file(session: dict) -> None:
    """Delete the temp image file from disk if it exists."""
    image_ref = session.get("image_data")
    if not image_ref or not isinstance(image_ref, dict):
        return
    path = image_ref.get("path")
    if path and os.path.exists(path):
        os.remove(path)
        logging.info(f"Cleaned up image: {path}")