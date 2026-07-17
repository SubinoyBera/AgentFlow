import re
from langchain_tavily import TavilySearch
from fastmcp import FastMCP
from src.logger import logging
import json
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("tavily-search")

@mcp.tool
async def tavily_search(query: str) -> str:
    """
    Web search tool using Tavily web-search API for searching the internet.
    Args:
        query (str): The query to search for.
    """
    try:
        tavily = TavilySearch(max_result=3, topic="general")
        response = tavily.invoke({"query": query})
        
        search_results = []
        for r in response.get("results", []):
            snippet = r["content"].replace("#", "").strip()
            snippet = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", snippet)
            snippet = re.sub(r"\[\]\([^)]+\)", "", snippet)
            
            search_results.append(
                {
                    "title": r.get("title", "None"), 
                    "url": r.get("url", "None"), 
                    "snippet": snippet
                }
            )
        
        logging.info("tavily_search tool called, and search results obtained")
        return json.dumps({
            "tavily_web_search_results": search_results
        })
    
    except Exception as e:
        logging.error(f"tavily_search tool failed: {e}")
        return json.dumps({
            "tavily_web_search_error": "Error during web search with tavily api"
        })

if __name__ == "__main__":
    mcp.run(transport="stdio")