import os
from fastmcp import FastMCP
from langchain_community.utilities import GoogleSerperAPIWrapper
from src.logger import logging
import json
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("news-search")

@mcp.tool
async def news_search(query: str) -> str:
    """
    News search tool using Serper API. Good for current news, politics, sports, tech, latest events.
    Args:
        query (str): Search query to search
    """
    try:
        serper_api = os.getenv("SERPER_API_KEY")
        search_client = GoogleSerperAPIWrapper(type="news", serper_api_key=serper_api)

        response = search_client.run(query=query)
        logging.info("news_search tool called, and search results obtained")
        
        return json.dumps({"news_results": response})
    
    except Exception as e:
        logging.error(f"news_search tool failed: {e}")
        return json.dumps({"news_search_error": "Error during news search with serper api"})

if __name__ == "__main__":
    mcp.run(transport="stdio")