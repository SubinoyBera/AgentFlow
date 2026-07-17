import os
import requests
from fastmcp import FastMCP
from src.logger import logging
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("calendar")

N8N_BASE_URL = os.getenv("N8N_BASE_URL")
REQUEST_TIMEOUT = 90

def _call_webhook(path: str, instruction: str) -> str:
    url = f"{N8N_BASE_URL}{path}"
    try:
        resp = requests.post(url, json={"message": instruction}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        try:
            data = resp.json()
            return data.get("response", data) if isinstance(data, dict) else str(data)
        
        except ValueError:
            return resp.text
    
    except requests.exceptions.Timeout:
        logging.error(f"n8n webhook timed out: {url}")
        return "Error: N8N service took too long to respond. Please try again in a moment."
    
    except requests.exceptions.RequestException as e:
        logging.error(f"n8n webhook call failed ({url}): {e}")
        return f"Error: could not complete this action ({str(e)}). N8N service may be unreachable."


@mcp.tool
async def calendar_agent(instruction: str) -> str:
    """
    Send a natural-language instruction to the n8n Calendar Agent. 
    It has full Calendar tool access (check availability, get events, create/update/delete event) 
    and decides which action to take purely from the instruction text.
    """
    return _call_webhook("/webhook/calendar-agent", instruction)


if __name__ == "__main__":
    mcp.run(transport="stdio")