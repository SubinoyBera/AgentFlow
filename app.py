import os, sys
import streamlit as st
import tempfile
from src.agent.langgraph_agent import ai_agent, external_kb_meta, checkpointer
from langchain_core.messages import HumanMessage
from src.pinecone.vectorstore import add_doc_to_vectorstore
from utils.common import retrieve_all_threads, generate_thread_id, load_pdf, clean_text, generate_summary
from src.logger.logging import logging
from src.exception.exception_handler import AppException


def reset_chat():
    """
    Resets the chat by generating a new thread ID and clearing the chat history.

    This function is called when the user wants to start a new conversation.
    It does not affect the chat threads stored in the session state.
    """
    try:
        new_thread_id = generate_thread_id()
        st.session_state["thread_id"] = new_thread_id
        add_thread(st.session_state["thread_id"])
        
        st.session_state["chat_history"] = []
        st.session_state["pinecone_index"] = False
        st.session_state["upload_key"] = st.session_state.get("upload_key", 0) + 1

    except Exception as e:
        logging.error(f"Error while resetting chat: {e}")
        raise AppException(e, sys)

def add_thread(thread_id):
    """
    Adds a new thread ID to the session state if it does not already exist.
    
    Args:
        thread_id (str): The thread ID to add to the session state.
    """
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_chat_conversations(thread_id):
    """
    Loads the chat conversations for the given thread ID.

    Args:
        thread_id (str): The thread ID to load the chat conversations for.

    Returns:
        list: A list of HumanMessage objects representing the chat conversations for the given thread ID.
    """
    try:
        return ai_agent.get_state(config={'configurable': {'thread_id': thread_id}}).values["messages"]      #type: ignore
    
    except Exception as e:
        logging.error(f"Error in loading chat conversations from thread_id: {e}")
        raise AppException(e, sys)


# ------------------------ Streamlit Session State -------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "pinecone_index" not in st.session_state:
    st.session_state["pinecone_index"] = False

if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads(checkpointer)

if "external_kb_meta" not in st.session_state:
    st.session_state["external_kb_meta"] = {
        "available": False,
        "topic": None,
        "summary": None
    }

add_thread(st.session_state["thread_id"])

# Configurable
CONFIG = {'configurable': {'thread_id': st.session_state["thread_id"]}}


# ------------------------ Main Chat UI ------------------------
if st.sidebar.button("New Chat ↗️"):
    reset_chat()

with st.sidebar:
    st.sidebar.header("➕ Upload PDF:")
    uploaded_file = st.file_uploader("Upload", type="pdf", key=st.session_state.get("upload_key")
)

    if uploaded_file is not None and not st.session_state["pinecone_index"]:
        with st.spinner("⏳ Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_path = tmp_file.name
 
            # Extract text from the pdf
            try:
                documents = load_pdf(temp_file_path)
                if documents:
                    extracted_text = "\n".join([doc.page_content for doc in documents])
                    extracted_text = clean_text(extracted_text)

                    summary = generate_summary(extracted_text)
                    # Upload document to vector store
                    add_doc_to_vectorstore(index_name=summary["topic"], content=extracted_text)

                    external_kb_meta["available"] = True
                    external_kb_meta["topic"] = summary["topic"]
                    external_kb_meta["summary"] = summary["summary"]
                    
                    # Update external_kb_meta in session state
                    st.session_state["external_kb_meta"].update({
                        "available": True,
                        "topic": summary["topic"],
                        "summary": summary["summary"]
                    })

                st.session_state["pinecone_index"] = True
                st.success("✅ Successfully uploaded!")
            
            except Exception as e:
                logging.error(f"Failed processing document: {e}")
                st.error("❌ Failed to process the uploaded file! Open 'New Chat' and please try again..")
                raise AppException(e, sys)

            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    uploaded_file = None

    st.sidebar.header("📂 Conversations:")
    for thread_id in st.session_state["chat_threads"][::-1]:
        conv_id = f"Chat-{str(thread_id)[:20]}"
        if st.sidebar.button(conv_id):
            st.session_state["thread_id"] = thread_id
            messages = load_chat_conversations(thread_id)

            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'

                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state["chat_history"] = temp_messages


# load converations
for msg in st.session_state["chat_history"]: 
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])  

user_input = st.chat_input("Ask anything ...")
   
if user_input:
    st.session_state["chat_history"].append({'role': 'user', 'content': user_input})
    with st.chat_message("human"):
        st.markdown(user_input)

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "external_kb_meta": external_kb_meta,
    }

    response = ai_agent.invoke(initial_state, config=CONFIG)       #type: ignore
    
    agent_message = response["messages"][-1].content
    st.session_state["chat_history"].append({'role': 'assistant', 'content': agent_message})
    
    with st.chat_message("assistant"):
        st.markdown(agent_message)
