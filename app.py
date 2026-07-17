import os, sys
import time
import streamlit as st
import tempfile
import markdown
from src.agent.runtime import get_runtime
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from utils import *
from app_styles import app_styles, STATUS_LABELS
from src.logger import logging
from src.exception.exception_handler import AppException

runtime = get_runtime()


# PAGE CONFIG + THEME
st.set_page_config(
    page_title="Agentic Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(app_styles, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def reset_chat():
    """
    Resets the chat by generating a new thread ID and clearing the chat history.
    Also clears any HITL pause and upload-processing flags so the next thread
    starts from a clean slate.
    """
    try:
        new_thread_id = generate_thread_id()
        st.session_state["thread_id"] = new_thread_id
        add_thread(st.session_state["thread_id"])

        st.session_state["chat_history"] = []
        st.session_state["image_processed"] = False
        st.session_state["pending_interrupt"] = None
        st.session_state["upload_key"] = st.session_state.get("upload_key", 0) + 1

    except Exception as e:
        logging.error(f"Error while resetting chat: {e}")
        raise AppException(e, sys)


def add_thread(thread_id):
    """Adds a new thread ID to the session state if it does not already exist."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_chat_conversations(thread_id):
    """Loads the chat conversations for the given thread ID."""
    try:
        state = runtime.get_state(config=get_config(thread_id))
        if not state or not state.values:
            return []
        return state.values.get("messages", [])

    except Exception as e:
        logging.error(f"Error in loading chat conversations from thread_id: {e}")
        raise AppException(e, sys)


def get_config(thread_id: str | None = None) -> RunnableConfig:
    tid = thread_id or st.session_state["thread_id"]
    return {
        'configurable': {'thread_id': tid},
        'metadata': {'thread_id': tid},
        'run_name': 'chat_turn',
    }


def get_active_interrupt(thread_id):
    """
    Checks the checkpointer directly for whether a given thread is CURRENTLY
    paused on a HITL approval. This matters because a user can switch away
    from a paused thread (via the sidebar) and come back later -- the pause
    lives in the checkpointer, not in st.session_state, so switching threads
    needs.
    """
    try:
        state = runtime.get_state(config=get_config(thread_id))
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None

    except Exception as e:
        logging.error(f"Error checking active interrupt for thread {thread_id}: {e}")
        return None


def extract_interrupt(result: dict):
    """
    LangGraph surfaces a pause as result['__interrupt__'] = [Interrupt(value=...), ...].
    Kept here for reference/tests; the live streaming path below checks chunk["__interrupt__"] inline instead.
    """
    interrupts = result.get("__interrupt__")
    if interrupts:
        return interrupts[0].value
    return None


def render_message(role: str, content: str, avatar: str | None = None):
    """
    Renders a single chat message in a Claude-style layout: a full-width flex
    row, with the bubble/prose aligned to the right for the user and to the
    left (no bubble chrome) for the assistant. Replaces the old column-split
    trick, which produced inconsistent widths and made the layout look
    disjointed.
    """
    avatar = avatar or ("🧑" if role == "user" else "🤖")
    # Convert markdown content to HTML using the 'extra' extension to support tables, fenced code blocks, etc.
    html_content = markdown.markdown(content, extensions=['extra'])
    st.markdown(
        f"""
        <div class="chat-row {role}">
            <div class="chat-avatar">{avatar}</div>
            <div class="chat-bubble">{html_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message_stream(role: str, stream_gen, avatar: str | None = None):
    """
    Streams the assistant's final answer with the same visual treatment as
    render_message. st.write_stream needs a real Streamlit container to
    write into, so we open the flex row as HTML, then let write_stream fill
    an inner placeholder, then close the row.
    """
    avatar = avatar or ("🧑" if role == "user" else "🤖")
    st.markdown(
        f"""<div class="chat-row {role}"><div class="chat-avatar">{avatar}</div>""",
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<div class="chat-bubble">', unsafe_allow_html=True)
        st.write_stream(stream_gen)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _extract_text(content) -> str:
    """Helper to safely extract string content from message content,
    which can be a string, a list of content blocks (dicts/strings), or None."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _typewriter(text: str, delay: float = 0.015):
    """Yields the text word-by-word so st.write_stream can reveal it
    progressively. NOTE: this is a UX polish over an already-complete string
    (answer_llm uses structured output, which returns its full payload at
    once) -- not literal token-by-token model streaming."""
    text_str = _extract_text(text)
    for word in text_str.split(" "):
        yield word + " "
        time.sleep(delay)


def run_agent_turn(graph_input):
    """
    Streams the graph turn-by-turn instead of blocking on a single .invoke().
    Shows a live status line for whichever node is currently executing, and
    reveals the final answer with a typewriter effect once it's ready.
    """
    interrupt_payload = None
    last_ai_content = None

    try:
        with st.status("Working on it...", expanded=True) as status:
            for chunk in get_runtime().stream(graph_input, config=CONFIG, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    interrupt_payload = chunk["__interrupt__"][0].value
                    break

                for node_name, node_update in chunk.items():
                    status.write(STATUS_LABELS.get(node_name, f"Running `{node_name}`..."))
                    if node_update and node_update.get("messages"):
                        last_ai_content = _extract_text(node_update["messages"][-1].content)

            status.update(
                label="Waiting for your approval" if interrupt_payload else "Done",
                state="running" if interrupt_payload else "complete",
            )

    except Exception as e:
        logging.error(f"Error streaming agent turn: {e}", exc_info=True)
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": "⚠️ Something went wrong while processing that. Please try again.",
        })
        st.session_state["pending_interrupt"] = None
        st.rerun()
        return

    if interrupt_payload is not None:
        st.session_state["pending_interrupt"] = interrupt_payload
        st.rerun()
        return

    st.session_state["pending_interrupt"] = None
    final_text = last_ai_content or "I don't have a response for that."

    render_message_stream("assistant", _typewriter(final_text))

    st.session_state["chat_history"].append({"role": "assistant", "content": final_text})
    st.rerun()


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "uploaded_doc" not in st.session_state:
    st.session_state["uploaded_doc"] = None

if "uploaded_doc_summary" not in st.session_state:
    st.session_state["uploaded_doc_summary"] = None

if "doc_processed" not in st.session_state:
    st.session_state["doc_processed"] = False

if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

if "image_data" not in st.session_state:
    st.session_state["image_data"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = runtime.retrieve_all_threads()

if "internal_kb_meta" not in st.session_state:
    st.session_state["internal_kb_meta"] = {
        "topic": "HR Policy",
        "summary": "Company overview Nexa AI and Leave Policies"
    }

if "image_processed" not in st.session_state:
    st.session_state["image_processed"] = False

# Tracks a paused HITL approval for the CURRENT thread. None = no pause active.
if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None


add_thread(st.session_state["thread_id"])
CONFIG = get_config()


# HEADER
st.markdown(
    """
    <div class="app-header"><h1>🤖 Multi-Agent AI Assistant</h1></div>
    <div class="app-subtitle">Research • Documents • Email — with human approval on anything that sends or schedules!</div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

with st.sidebar:
    st.divider()
    st.subheader("📎 Upload")
    uploaded_file = st.file_uploader(
        "PDF or image",
        type=["pdf", "png", "jpg", "jpeg"],
        key=st.session_state.get("upload_key"),
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        # PDF PROCESSING
        if file_extension == "pdf" and not st.session_state.get("doc_processed"):
            with st.spinner("⏳ Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_file_path = tmp_file.name

                try:
                    documents = load_pdf(temp_file_path)
                    if documents:
                        extracted_text = "\n".join([doc.page_content for doc in documents])
                        extracted_text = clean_text(extracted_text)
                        summary = generate_summary(extracted_text)

                        st.session_state["uploaded_doc"] = extracted_text
                        st.session_state["uploaded_doc_summary"] = summary
                        st.session_state["doc_processed"] = True

                    st.success("✅ Document ready")

                except Exception as e:
                    logging.error(f"Failed to process uploaded document: {e}", exc_info=True)
                    st.error("❌ Failed to process the file. Start a new chat and try again.")
                    raise AppException(e, sys)

                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

        # IMAGE PROCESSING
        elif file_extension in ["png", "jpg", "jpeg"] and not st.session_state["image_processed"]:
            try:
                with st.spinner("⏳ Processing image..."):
                    image_data = prepare_image_data(uploaded_file)
                    st.session_state["image_data"] = image_data
                    st.session_state["image_processed"] = True

                st.success("✅ Image ready")

            except Exception as e:
                logging.error(f"Failed to process image: {e}", exc_info=True)
                st.error("❌ Failed to process the image. Start a new chat and try again.")
                raise AppException(e, sys)

    st.divider()
    st.subheader("💬 Conversations")
    for thread_id in st.session_state["chat_threads"][::-1]:
        conv_label = f"Chat · {str(thread_id)[:8]}"
        if st.button(conv_label, key=f"thread_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            messages = load_chat_conversations(thread_id)

            temp_messages = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": _extract_text(msg.content)})

            st.session_state["chat_history"] = temp_messages

            # Re-discover whether THIS thread is sitting on a HITL pause --
            # important if the user left it mid-approval and is coming back.
            st.session_state["pending_interrupt"] = get_active_interrupt(thread_id)
            st.rerun()

    st.divider()
    st.caption(f"Thread: `{str(st.session_state['thread_id'])[:12]}...`")


# MAIN CHAT AREA -- transcript
for msg in st.session_state["chat_history"]:
    render_message(msg["role"], msg["content"])


# MAIN CHAT AREA -- either a pending HITL approval, or the normal chat input
if st.session_state["pending_interrupt"] is not None:
    payload = st.session_state["pending_interrupt"]

    st.markdown('<div class="hitl-card">', unsafe_allow_html=True)
    st.markdown('<div class="hitl-title">✋ Approval needed</div>', unsafe_allow_html=True)
    st.write(payload.get("message", "This action requires your approval before I proceed."))

    # answer_node's interrupt payload includes "draft"; workspace_agent's
    # includes "pending_action". Render whichever is present.
    if payload.get("draft"):
        st.text_area("Drafted content", payload["draft"], height=160, disabled=True)
    if payload.get("pending_action"):
        with st.expander("Action details"):
            st.json(payload["pending_action"])

    feedback_text = st.text_input(
        "If you reject, what should change? (optional)",
        key="rejection_feedback",
        placeholder="e.g. move it to Sunday afternoon instead",
    )

    col1, col2 = st.columns(2)
    approve_clicked = col1.button("✅ Approve", use_container_width=True, type="primary")
    reject_clicked = col2.button("❌ Reject", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if approve_clicked:
        st.session_state["chat_history"].append({"role": "user", "content": "✅ Approved"})
        run_agent_turn(Command(resume={"approved": True, "feedback": None}))

    if reject_clicked:
        note = feedback_text.strip() or "Rejected without additional feedback."
        st.session_state["chat_history"].append({"role": "user", "content": f"❌ Rejected — {note}"})
        run_agent_turn(Command(resume={"approved": False, "feedback": feedback_text.strip() or None}))

else:
    user_input = st.chat_input("Ask anything...")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        render_message("user", user_input)

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "query": user_input,
            "internal_kb_meta": st.session_state["internal_kb_meta"],
            "uploaded_doc": st.session_state["uploaded_doc"],
            "uploaded_doc_summary": st.session_state["uploaded_doc_summary"],
            "image_data": st.session_state["image_data"],
        }

        run_agent_turn(initial_state)