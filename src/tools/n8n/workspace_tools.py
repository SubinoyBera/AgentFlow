"""
workspace_automation.py
"""
import os
import requests
from langchain_core.tools import tool
from src.logger import logging

N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://awaited-bunny-multiply.ngrok-free.app")
REQUEST_TIMEOUT = 90  # seconds


def _call_webhook(path: str, instruction: str) -> str:
    url = f"{N8N_BASE_URL}{path}"
    try:
        resp = requests.post(url, json={"message": instruction}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        try:
            data = resp.json()
            # Adjust the "output" key if your n8n "Respond to Webhook" node wraps things differently.
            return data.get("response", data) if isinstance(data, dict) else str(data)
        except ValueError:
            return resp.text
    except requests.exceptions.Timeout:
        logging.error(f"n8n webhook timed out: {url}")
        return "Error: the automation service took too long to respond. Please try again in a moment."
    except requests.exceptions.RequestException as e:
        logging.error(f"n8n webhook call failed ({url}): {e}")
        return f"Error: could not complete this action ({str(e)}). The automation service may be unreachable."


@tool
def gmail_agent(instruction: str) -> str:
    """
    Send a natural-language instruction to the n8n Email Agent. It has full Gmail tool access
    (get emails, get labels, add labels, reply in thread, draft email, mark read, send email) and
    decides which action(s) to take purely from the instruction text -- there is no structural
    restriction on what it can do. Callers MUST scope instructions precisely: strictly read-only
    phrasing ("only fetch, do not send/reply/delete anything") while gathering context, and an
    explicit single-action instruction ("use ONLY the Send Email tool, send exactly this...") when
    actually sending -- never the user's raw request verbatim.
    """
    return _call_webhook("/webhook/email-agent", instruction)


@tool
def calendar_agent(instruction: str) -> str:
    """
    Send a natural-language instruction to the n8n Calendar Agent. It has full Calendar tool
    access (check availability, get events, create/update/delete event) and decides which action to
    take purely from the instruction text. Same caveat as gmail_agent: callers MUST scope
    instructions precisely -- read-only phrasing while gathering context, an explicit single-action
    instruction only after approval.
    """
    return _call_webhook("/webhook/calendar-agent", instruction)