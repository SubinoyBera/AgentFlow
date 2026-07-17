import os, time
import requests
from fastmcp import FastMCP
from src.logger import logging
import json
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("stock-finance")

@mcp.tool
async def stock_finance_tool(symbol: str) -> str:                                                                       #type: ignore
    """
    Stock finance tool to get the latest stock data of a given company symbol.
    """
    symbol = symbol.split(":")[-1].strip(' "{}')
    stock_finance_api = os.getenv("STOCK_FINANCE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_finance_api}"

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            data = requests.get(url).json()
        
        except Exception as e:
            logging.error(f"stock_finance_tool request failed (attempt {attempt}): {e}")
            return json.dumps({
                "stock_finance_tool_error": "Failed to get data"
            })
        
        if data.get("Global Quote"):
            logging.info("stock_finance_tool called, and tool results obtained")
            return json.dumps(data)
        
        if attempt < max_attempts:
            logging.warning(f"stock_finance_tool rate-limited for {symbol} (attempt {attempt}), retrying...")
            time.sleep(1.5 * attempt)
            continue
        
        logging.error(f"stock_finance_tool got no usable data for {symbol}: {data}")
        return json.dumps({
            "stock_finance_tool_error": f"No stock data available for '{symbol}': {data}"
        })


if __name__ == "__main__":
    mcp.run(transport="stdio")