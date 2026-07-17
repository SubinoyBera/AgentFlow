import os
import requests
from fastmcp import FastMCP
from src.logger import logging
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("gmail")

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
        return "Error: the automation service took too long to respond. Please try again in a moment."
    
    except requests.exceptions.RequestException as e:
        logging.error(f"n8n webhook call failed ({url}): {e}")
        return f"Error: could not complete this action ({str(e)}). The automation service may be unreachable."


@mcp.tool
async def gmail_agent(instruction: str) -> str:
    """
    Send a natural-language instruction to the n8n Email Agent. 
    It has full Gmail tool access (get emails, get labels, add labels, reply in thread, draft email, mark read, send email) 
    and decides which action(s) to take purely from the instruction text -- there is no structural restriction on what it can do. 
    Callers MUST scope instructions precisely.
    """
    return _call_webhook("/webhook/email-agent", instruction)


if __name__ == "__main__":
    mcp.run(transport="stdio")