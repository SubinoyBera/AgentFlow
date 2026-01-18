import sqlite3
import uuid
import re
from datetime import datetime, timezone, timedelta
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langsmith import traceable
from pydantic import BaseModel, Field
from src.agent.prompt import doc_summarizer_prompt
from src.logger.logging import logging


def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def get_checkpointer():
    conn = sqlite3.connect(database='data/chat_datastore.db', check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    return checkpointer


def retrieve_all_threads(checkpointer):
    all_threads =  set()
    for checkpoint in checkpointer.list(None):
        configurable = checkpoint.config.get("configurable")
        if configurable and "thread_id" in configurable:
            all_threads.add(configurable["thread_id"])

    return list(all_threads)


@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load() 


def clean_text(text: str) -> str:
    # Normalize unicode
    text = text.replace("\u00a0", " ")  # non-breaking space

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim spaces around newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    return text.strip()


def covert_to_exact_time(sunrise_utc, sunset_utc, tz_offset, dt_utc):
    local_tz = timezone(timedelta(seconds=tz_offset))
    sunrise_local = datetime.fromtimestamp(sunrise_utc, tz=local_tz)
    sunset_local = datetime.fromtimestamp(sunset_utc, tz=local_tz)
    dt_local = datetime.fromtimestamp(dt_utc, tz=local_tz)

    return sunrise_local.strftime("%H:%M:%S"), sunset_local.strftime("%H:%M:%S"), dt_local.strftime("%Y-%m-%d %H:%M:%S")


class DocSummerizerResponse(BaseModel):
    topic: str = Field(description="An appropriate topic for the document")
    summary: str = Field(description="Summary of the document")


def generate_summary(doc: str) -> dict:
    """
    Generate a summary of the given document using LLM

    Args:
        doc (str): The document to be summarized

    Returns:
        str : The generated summary of the document
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(DocSummerizerResponse)
    prompt = ChatPromptTemplate(
        [
            ("system", "You are an expert writer"),
            ("human", doc_summarizer_prompt)
        ],
        input_variables = ["doc"]
    )
    chain = prompt | llm
    response : DocSummerizerResponse = chain.invoke({"doc": doc})       # type: ignore

    return {"topic": response.topic, "summary": response.summary}