import os
import re
import requests
import wikipedia
from typing import Annotated
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_tavily import TavilySearch
from src.pinecone.vectorstore import get_retriever
from utils.common import covert_to_exact_time
from src.logger.logging import logging

@tool
def retriever(query: str, index_name: str, reranker):
    """
    A Langchain tool that retrieves documents from a knowledge base using MMR and reranks using a given reranker.
    
    Args:
        query (str): The query to search for in the knowledge base.
        index_name (str): The name of the Pinecone index to search in.
        reranker: The reranker to use for reranking the retrieved documents.
    
    Returns:
        str: The content of the top 4 retrieved documents, separated by newlines.
    """
    try:
        logging.info("Retrieving docs from vectorstore")
        retriever_instance = get_retriever(index_name)
        # MMR retriever
        docs = retriever_instance.invoke(
            query,
            search_type="mmr",
            k=10,
            lambda_mult=0.5
        )

        if not docs:
            logging.warning("Retriever did not fetch any doc!")
            return None
        
        # Prepare for reranking
        pairs = [(query, d.page_content) for d in docs]

        logging.info("Reranking docs")
        scores = reranker.predict(pairs)

        # Combine scores with docs
        scored_docs = list(zip(docs, scores))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Threshold filtering
        threshold_filtered_docs = [
            doc for doc, score in scored_docs if score >= 0.4
        ]

        if not threshold_filtered_docs:
            logging.warning("All reranker scores below reranker threshold")
            return None  
    
        final_docs = threshold_filtered_docs[:4]
        logging.info("Done")
        return "\n\n".join(d.page_content for d in final_docs)

    except Exception as e:
        logging.error(f"Langchain retriever tool failed: {e}")
        return None


@tool
def tavily_search(query: str) -> dict:
    """
    Web search tool using Tavily web-search API for searching the internet.

    Args:
        query (str): The query to search for.

    Returns:
        dict: A dictionary containing the web search results.

    """
    try:
        tavily = TavilySearch(max_result=3, topic="general")
        response = tavily.invoke({"query": query})
        search_results = []
        for r in response.get("results", []):
            snippet = r["content"].replace("#", "").strip()
            # Remove markdown links: [text](url)
            snippet = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", snippet)
            # Remove empty links: []()
            snippet = re.sub(r"\[\]\([^)]+\)", "", snippet)
            search_results.append({
                "title": r.get("title", "None"),
                "url": r.get("url", "None"),
                "snippet": snippet
            })

        logging.info("tavily_search tool called, and search results obtained")
        return {"tavily_web_search_results": search_results}
        
    except Exception as e:
        logging.error(f" tavily_search tool failed: {e}")
        return {"tavily_web_search_error": "Error during web search with tavily api"}
    

@tool
def news_search(query: str) -> dict :
    """
    News search tool using Serper API. 
    Provides good search results about current news in politics, sports, technology, latest events, etc.

    Args:
        query (str) : Search query to search

    Returns :
        dict: A dictionary containing web search results
    """
    try:
        serper_api = os.getenv("SERPER_API_KEY")
        search_client = GoogleSerperAPIWrapper(type="news", serper_api_key=serper_api)
        response = search_client.run(query=query)

        logging.info("news_search tool called, and search results obtained")
        return {"news_results": response}
    
    except Exception as e:
        logging.error(f"news_search tool failed: {e}")
        return {"news_search_error": "Error during news search with serper api"}


@tool
def wiki_search(query: str):
    """
    Wikipedia search tool.
    Useful for retrieving concise encyclopedic background information, historical facts, definitions, 
    biographies, scientific concepts, and general knowledge from Wikipedia.
    """
    try:
        wiki_results = wikipedia.summary(query, sentences=30)
        logging.info("wiki_search tool called, and search results obtained")
        return {"wiki_results": wiki_results}
    
    except Exception as e:
        logging.error(f"wiki_search tool failed: {e}")
        return {"wiki_search_error": "Error during wikipedia search"}


@tool
def weather_tool(location: str) -> dict:
    """
    Weather tool to get current weather report of any given location.

    location : str
        Name of the location to get the weather report

    Returns:
        dict: A dictionary containing the weather report of the location
    """
    try:
        weather_api = os.getenv("WEATHER_API_KEY")
        loc = location.split(":")[-1].strip(' "{}')
        # get latitude longitude of the location
        lat_long_url = f"http://api.openweathermap.org/geo/1.0/direct?q={loc}&limit=1&appid={weather_api}"
        response = requests.get(lat_long_url).json()
        lat = response[0]['lat']
        lon = response[0]['lon']
        # get weather
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_api}&units=metric"
        weather = requests.get(weather_url).json()

        sunrise_utc, sunset_utc, tz_offset, dt_utc = weather["sys"]["sunrise"], weather["sys"]["sunset"], weather["timezone"], weather["dt"]
        sunrise, sunset, report_time = covert_to_exact_time(sunrise_utc, sunset_utc, tz_offset, dt_utc)

        report = {
            "main": weather["weather"][0]["main"],
            "description": weather["weather"][0]["description"],
            "conditions": weather["main"],
            "visibility": weather["visibility"],
            "wind": weather["wind"],
            "clouds": weather["clouds"],
            "extras": {"sunrise": sunrise, "sunset": sunset, "report_time": report_time, "country": weather["sys"]["country"]}   
        }
        
        logging.info("weather_tool called, and tool results obtained")
        return report
    
    except Exception as e:
        logging.error(f"weather_tool failed: {e}")
        return {"weather_tool_error": "Weather tool failed to get weather report"}


@tool
def stock_finance_tool(symbol: Annotated[str, "Symbol for the company whose stock data is to be fetched"]) -> dict: 
    """
    Stock finance tool to get the latest stock data of a given company symbol.
    Provides best results about current stock price data, and latest updates in stocks about the company.
    Example symbols: 'AAPL' for Apple, 'NVDA' for Nvidia, 'AMZN' for Amazon, etc..

    Args: 
        symbol (str): Symbol for the company whose stock data is to be fetched

    Returns:
        dict: A dictionary containing the stock data of the symbol
    """
    try: 
        symbol = symbol.split(":")[-1].strip(' "{}')
        stock_finance_api = os.getenv("STOCK_FINANCE_API_KEY")
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_finance_api}'
        data = requests.get(url).json()

        logging.info("stock_finance_tool called, and tool results obtained")
        return data

    except Exception as e:
        logging.error(f"stock_finance_tool failed: {e}")
        return {"stock_finance_tool": "Failed to get data"}