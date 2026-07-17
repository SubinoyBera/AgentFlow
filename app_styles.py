app_styles = """
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

    /* Markdown Formatting inside chat bubbles */
    .chat-bubble h1, .chat-bubble h2, .chat-bubble h3, .chat-bubble h4 {
        color: #ffffff;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .chat-bubble h1 { font-size: 1.3rem; }
    .chat-bubble h2 { font-size: 1.15rem; }
    .chat-bubble h3 { font-size: 1.05rem; }

    .chat-bubble ul, .chat-bubble ol {
        margin-top: 0.3rem;
        margin-bottom: 0.8rem;
        padding-left: 1.5rem;
    }
    .chat-bubble li {
        margin-bottom: 0.25rem;
    }

    .chat-bubble table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    .chat-bubble th, .chat-bubble td {
        border: 1px solid #2E3140;
        padding: 6px 10px;
        text-align: left;
    }
    .chat-bubble th {
        background-color: #1B1D28;
        color: #ffffff;
    }

    .chat-bubble pre {
        background-color: #161822;
        border: 1px solid #2E3140;
        border-radius: 6px;
        padding: 0.75rem;
        overflow-x: auto;
        margin: 0.8rem 0;
    }
    .chat-bubble code {
        font-family: Consolas, Monaco, 'Andale Mono', monospace;
        font-size: 0.88rem;
        background-color: #1B1D28;
        padding: 2px 4px;
        border-radius: 4px;
        color: #40C4FF;
    }
    .chat-bubble pre code {
        background-color: transparent;
        padding: 0;
        color: #E4E6EB;
    }

    .chat-bubble blockquote {
        border-left: 3px solid #7C5CFC;
        margin: 0.5rem 0;
        padding-left: 0.8rem;
        color: #8A8F98;
        font-style: italic;
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
"""

STATUS_LABELS = {
    "assistant": "🧭 Understanding your request...",
    "supervisor": "🧠 Planning the next step...",
    "research_agent": "🔍 Researching...",
    "workspace_agent": "📋 Preparing for action...",
    "workspace_confirm": "📬 Finalizing action...",
    "answer_agent": "✍️ Drafting a response...",
    "answer_confirm": "✅ Finalizing the response...",
    "vision": "🖼️ Looking at your image...",
}