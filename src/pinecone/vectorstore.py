import os, sys
from dotenv import load_dotenv
load_dotenv()
import asyncio
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable
from pydantic import SecretStr
from src.logger.logging import logging
from src.exception.exception_handler import AppException

# initialize pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
client = Pinecone(api_key=pinecone_api_key)

def get_embeddings():
    """
    Returns a embedding object that uses the GoogleGenerativeAIEmbeddings model to generate embeddings for vectors.
    """
    try:
        # Ensure a running event loop exists for Streamlit threads
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key is not None:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=SecretStr(google_api_key))
            return embeddings
        else:
            raise EnvironmentError("Google API key environment variable not found")
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {e}")
    

def create_pinecone_index(index_name: str):
    """
    Creates a Pinecone index with the given name if it doesn't already exist.

    Args:
        index_name (str): The name of the Pinecone index to create.

    Raises:
        AppException: If the index creation fails.
    """
    if index_name not in client.list_indexes().names():
        try:
            client.create_index(
                name = index_name,
                dimension = 768,
                metric = "cosine",
                spec = ServerlessSpec(cloud='aws', region='us-east-1')
            )
            logging.info("Pinecone index created successfully")

        except Exception as e:
            logging.error(f"Failed to create pinecone index: {e}")
            raise AppException(e, sys)


# retriever function
def get_retriever(index_name: str):
    """
    Retrieves a LangChain retriever object from a Pinecone index.
    Args:
        index_name (str): The name of the Pinecone index.

    Returns:
        PineconeVectorStore: langChain retriever object for Pinecone Vectorstore.
    """
    if index_name.lower() not in client.list_indexes().names():
        create_pinecone_index(index_name.lower())
        
    index = client.Index(index_name.lower())
    vectorstore = PineconeVectorStore(index=index, embedding=get_embeddings())
    return vectorstore.as_retriever()


# upload document to vector store
@traceable(name="upload_doc_to_pinecone")
def add_doc_to_vectorstore(index_name: str, content: str):
    """
    Adds a document to the Pinecone vector store.

    The document is split into chunks before being added to the vector store. 
    The chunks are created using the RecursiveCharacterTextSplitter from Langchain.

    Args:
        index_name (str): The name of the Pinecone index
        content (str): The content to added to the vector store.
    """
    if content is None:
        raise ValueError("No content found to add in the vector store")
    
    if index_name.lower() not in client.list_indexes().names():
        create_pinecone_index(index_name.lower())

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            separators=["\n\n", "\n", ".", " "],
            add_start_index = True
        )
        # create langchain document object
        doc = text_splitter.create_documents(texts=[content])

        index = client.Index(index_name.lower())
        vectorstore = PineconeVectorStore(index=index, embedding=get_embeddings())

        # add documents to vector store
        vectorstore.add_documents(doc)
        logging.info("Uploaded document chunks to pinecone vectore store successfully")

    except Exception as e:
        logging.error(f"Error during document chucking and uploading in pinecone vector store: {e}")
        raise AppException(e, sys)