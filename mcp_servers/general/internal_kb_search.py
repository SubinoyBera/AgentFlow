from fastmcp import FastMCP
from pathlib import Path
from sentence_transformers import CrossEncoder
from src.db_connections.pinecone.vectorstore import get_retriever
from src.logger import logging

mcp = FastMCP("internal-kb-search")

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        logging.info("Loading reranker model (first use)...")
        _reranker = CrossEncoder(str(Path("models/bge-reranker-base")))
    
    return _reranker


@mcp.tool
async def internal_kb_search(query: str, index_name: str) -> str:
    """
    Search the INTERNAL business knowledge base (RAG) for relevant documents.
    This is NOT for user-uploaded PDFs -- those are handled separately, never via this tool.

    Args:
        query: The search query.
        index_name: The exact internal knowledge-base index name to search --
            use the exact value given to you in your system prompt/instructions,
            never guess or invent one.
    """
    try:
        reranker = _get_reranker()
        logging.info("Retrieving docs from vectorstore")
        retriever_instance = get_retriever(index_name)
        docs = retriever_instance.invoke(query, search_type="mmr", k=10, lambda_mult=0.5)

        if not docs:
            logging.warning("Retriever did not fetch any doc!")
            return "No relevant documents found in internal knowledge base."

        pairs = [(query, d.page_content) for d in docs]
        logging.info("Reranking docs")
        scores = reranker.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        threshold_filtered_docs = [doc for doc, score in scored_docs if score >= 0.4]
        if not threshold_filtered_docs:
            logging.warning("All reranker scores below reranker threshold")
            return "No relevant documents found in internal knowledge base."

        final_docs = threshold_filtered_docs[:4]
        logging.info("Done")
        return "\n\n".join(d.page_content for d in final_docs)

    except Exception as e:
        logging.error(f"internal_kb_search tool failed: {e}")
        return "Error retrieving documents from the internal knowledge base."


if __name__ == "__main__":
    mcp.run(transport="stdio")