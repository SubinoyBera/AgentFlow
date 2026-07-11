"""
workspace_automation.py

Four tools instead of two multiplexed ones:
- gmail_read_tool / calendar_read_tool  -> hit the READ-ONLY n8n AI Agent webhooks. Safe to bind
  to a freeform tool-calling agent since nothing behind these can send/create/delete anything.
- gmail_send_tool / calendar_write_tool -> hit the DETERMINISTIC n8n webhooks (no AI Agent node on
  the n8n side). Only ever call these after explicit human approval, with fully-specified fields --
  there's no LLM on the n8n end to "interpret" what you meant.

N8N_BASE_URL comes from the environment instead of being hardcoded, since free-tier ngrok URLs
rotate on every restart -- update the .env, not the code.
"""
import os
from typing import Optional
import requests
from langchain_core.tools import tool
from src.logger.logging import logging

N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://awaited-bunny-multiply.ngrok-free.app")
REQUEST_TIMEOUT = 30


def _call_webhook(path: str, payload: dict) -> str:
    """
    Shared POST helper. Always returns a string (the tool result the LLM sees), never raises --
    errors come back as a descriptive string so the calling agent can react instead of crashing.
    """
    url = f"{N8N_BASE_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        try:
            data = resp.json()
            return data.get("output", data) if isinstance(data, dict) else str(data)

        except ValueError:
            return resp.text

    except requests.exceptions.Timeout:
        logging.error(f"n8n webhook timed out: {url}")
        return "Error: the automation service took too long to respond. Please try again in a moment."

    except requests.exceptions.RequestException as e:
        logging.error(f"n8n webhook call failed ({url}): {e}")
        return f"Error: could not complete this action ({str(e)}). The automation service may be unreachable."


@tool
def gmail_read_tool(instruction: str) -> str:
    """
    Look up Gmail data ONLY -- e.g. get the latest email, search by sender/subject, get labels,
    read a thread. This hits an n8n workflow with no send/reply/draft/delete capability at all, so
    it is safe to call freely while gathering context. Pass a natural-language description of what
    to look up (e.g. "get the most recent email in the inbox" or "find emails from john@company.com
    about the Q3 report").
    """
    return _call_webhook("/webhook/email-agent-read", {"message": instruction})


@tool
def calendar_read_tool(instruction: str) -> str:
    """
    Look up calendar data ONLY -- e.g. check availability for a time window, list upcoming events.
    This hits an n8n workflow with no create/update/delete capability at all, so it is safe to call
    freely while gathering context. Pass a natural-language description of what to look up.
    """
    return _call_webhook("/webhook/calendar-agent-read", {"message": instruction})


@tool
def gmail_send_tool(to: str, subject: str, body: str, thread_id: Optional[str] = None) -> str:
    """
    Send an email exactly as specified. If thread_id is provided, this replies within that thread
    instead of starting a new email. Only call this after the user has explicitly approved the
    exact recipient, subject, and body -- this executes immediately with no further confirmation.
    """
    payload = {"to": to, "subject": subject, "body": body}
    if thread_id:
        payload["thread_id"] = thread_id
    return _call_webhook("/webhook/email-send", payload)


@tool
def calendar_write_tool(action: str, event_details: dict) -> str:
    """
    Create, update, or delete a calendar event exactly as specified. `action` must be one of
    'create', 'update', 'delete'. `event_details` should contain whatever fields that action needs
    (title, start, end, attendees, location, event_id for update/delete, etc.). Only call this
    after the user has explicitly approved the exact event details -- this executes immediately
    with no further confirmation.
    """
    if action not in ("create", "update", "delete"):
        return f"Error: invalid action '{action}'. Must be 'create', 'update', or 'delete'."
    payload = {"action": action, **(event_details or {})}
    return _call_webhook("/webhook/calendar-write", payload)