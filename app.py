import os, sys
import time
import streamlit as st
import tempfile
from src.agent.workflow import ai_agent, checkpointer
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from utils import *
from src.logger import logging
from src.exception.exception_handler import AppException


# ---------------------------------------------------------------------------
# PAGE CONFIG + THEME
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ---- overall page ---- */
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #12141c 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 7rem;
        max-width: 820px;
    }

    /* ---- header ---- */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.2rem;
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7C5CFC, #40C4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .app-subtitle {
        color: #8A8F98;
        font-size: 0.92rem;
        margin-bottom: 2rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid #22242f;
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #14151d;
        border-right: 1px solid #262835;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
        color: #B5B9C6 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ---- buttons ---- */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #2E3140;
        background: #1B1D28;
        color: #E4E6EB;
        transition: all 0.15s ease;
    }
    .stButton>button:hover {
        border-color: #7C5CFC;
        color: #ffffff;
    }

    /* ---- chat rows (Claude-style: full-width row, content aligned by role) ---- */
    .chat-row {
        display: flex;
        width: 100%;
        margin-bottom: 1.35rem;
        gap: 0.65rem;
    }
    .chat-row.user {
        justify-content: flex-end;
    }
    .chat-row.assistant {
        justify-content: flex-start;
    }

    .chat-avatar {
        flex-shrink: 0;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        background: #1B1D28;
        border: 1px solid #2E3140;
    }
    .chat-row.user .chat-avatar {
        order: 2;
    }

    .chat-bubble {
        max-width: 72%;
        font-size: 0.96rem;
        line-height: 1.55;
        color: #E4E6EB;
    }
    /* User: a real bubble, pill-shaped, right aligned */
    .chat-row.user .chat-bubble {
        background: linear-gradient(135deg, #2A2550, #241E45);
        border: 1px solid #3A3268;
        border-radius: 16px 16px 4px 16px;
        padding: 0.65rem 1rem;
    }
    /* Assistant: no bubble chrome, reads like plain prose (Claude-style) */
    .chat-row.assistant .chat-bubble {
        background: transparent;
        border: none;
        padding: 0.15rem 0;
        max-width: 100%;
    }
    .chat-bubble p {
        margin: 0 0 0.5rem 0;
    }
    .chat-bubble p:last-child {
        margin-bottom: 0;
    }

    /* ---- HITL approval card ---- */
    .hitl-card {
        border: 1px solid #7C5CFC55;
        background: #1A1730;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 0.4rem 0 1rem 0;
    }
    .hitl-title {
        color: #C9B8FF;
        font-weight: 600;
        font-size: 0.98rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ---- chat input ---- */
    div[data-testid="stChatInput"] {
        max-width: 820px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)


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
        return ai_agent.get_state(config={'configurable': {'thread_id': thread_id}}).values["messages"]  # type: ignore

    except Exception as e:
        logging.error(f"Error in loading chat conversations from thread_id: {e}")
        raise AppException(e, sys)


def get_config(thread_id: str | None = None):
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
    needs to re-discover it rather than assume it's gone.
    """
    try:
        state = ai_agent.get_state(config=get_config(thread_id))
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None
    except Exception as e:
        logging.error(f"Error checking active interrupt for thread {thread_id}: {e}")
        return None


def extract_interrupt(result: dict):
    """LangGraph surfaces a pause as result['__interrupt__'] = [Interrupt(value=...), ...].
    Kept here for reference/tests; the live streaming path below checks
    chunk["__interrupt__"] inline instead."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        return interrupts[0].value
    return None


# ---------------------------------------------------------------------------
# Friendly labels shown while each graph node is actively running. Keys must
# match your node names exactly (see graph.add_node(...) in workflow.py).
# ---------------------------------------------------------------------------
STATUS_LABELS = {
    "assistant": "🧭 Understanding your request...",
    "supervisor": "🧠 Planning the next step...",
    "research_agent": "🔍 Researching...",
    "workspace_agent": "📋 Preparing the calendar/email action...",
    "workspace_confirm": "📬 Finalizing the workspace action...",
    "answer_agent": "✍️ Drafting a response...",
    "answer_confirm": "✅ Finalizing the response...",
    "vision": "🖼️ Looking at your image...",
}


def render_message(role: str, content: str, avatar: str | None = None):
    """
    Renders a single chat message in a Claude-style layout: a full-width flex
    row, with the bubble/prose aligned to the right for the user and to the
    left (no bubble chrome) for the assistant. Replaces the old column-split
    trick, which produced inconsistent widths and made the layout look
    disjointed.
    """
    avatar = avatar or ("🧑" if role == "user" else "🤖")
    # st.markdown escapes nothing by itself, so we rely on content already
    # being plain text/markdown-safe (as it was previously).
    safe_content = content.replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="chat-row {role}">
            <div class="chat-avatar">{avatar}</div>
            <div class="chat-bubble"><p>{safe_content}</p></div>
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


def _typewriter(text: str, delay: float = 0.015):
    """Yields the text word-by-word so st.write_stream can reveal it
    progressively. NOTE: this is a UX polish over an already-complete string
    (answer_llm uses structured output, which returns its full payload at
    once) -- not literal token-by-token model streaming."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def run_agent_turn(graph_input):
    """
    Streams the graph turn-by-turn instead of blocking on a single .invoke().
    Shows a live status line for whichever node is currently executing, and
    reveals the final answer with a typewriter effect once it's ready.
    Handles being interrupted mid-stream by a HITL pause exactly like the
    old invoke()-based version did.
    """
    interrupt_payload = None
    last_ai_content = None

    try:
        with st.status("Working on it...", expanded=True) as status:
            for chunk in ai_agent.stream(graph_input, config=CONFIG, stream_mode="updates"):  # type: ignore
                if "__interrupt__" in chunk:
                    interrupt_payload = chunk["__interrupt__"][0].value
                    break

                for node_name, node_update in chunk.items():
                    status.write(STATUS_LABELS.get(node_name, f"Running `{node_name}`..."))
                    if node_update and node_update.get("messages"):
                        last_ai_content = node_update["messages"][-1].content

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

if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

if "image_data" not in st.session_state:
    st.session_state["image_data"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads(checkpointer)

if "external_kb_meta" not in st.session_state:
    st.session_state["external_kb_meta"] = {}


if "image_processed" not in st.session_state:
    st.session_state["image_processed"] = False

# Tracks a paused HITL approval for the CURRENT thread. None = no pause active.
if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

add_thread(st.session_state["thread_id"])

CONFIG = get_config()


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
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
        if file_extension == "pdf":
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
                        st.session_state["pinecone_index"] = True  # FIX: mark as done

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
                    st.session_state["image_processed"] = True  # FIX: mark as done

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
                temp_messages.append({"role": role, "content": msg.content})

            st.session_state["chat_history"] = temp_messages

            # Re-discover whether THIS thread is sitting on a HITL pause --
            # important if the user left it mid-approval and is coming back.
            st.session_state["pending_interrupt"] = get_active_interrupt(thread_id)
            st.rerun()

    st.divider()
    st.caption(f"Thread: `{str(st.session_state['thread_id'])[:12]}...`")


# ---------------------------------------------------------------------------
# MAIN CHAT AREA -- transcript
# ---------------------------------------------------------------------------
for msg in st.session_state["chat_history"]:
    render_message(msg["role"], msg["content"])


# ---------------------------------------------------------------------------
# MAIN CHAT AREA -- either a pending HITL approval, or the normal chat input
# ---------------------------------------------------------------------------
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
            "external_kb_meta": st.session_state["external_kb_meta"],
            "uploaded_doc": st.session_state["uploaded_doc"],
            "uploaded_doc_summary": st.session_state["uploaded_doc_summary"],
            "image_data": st.session_state["image_data"],
        }

        run_agent_turn(initial_state)