# ---------------------------------------------------------------------------
# N8N TOOL CATALOG
# ---------------------------------------------------------------------------

"""
Each entry names an n8n tool already exposed inside the multiplexed n8n agent (gmail_agent / calender_agent are single "instruction in, agent decides everything" webhookscthat internally expose several named tools -- see your n8n canvas). Adding a new capability means
adding ONE entry here (and, for a new domain, one line in DOMAIN_AGENTS) -- no new Python function, or graph changes.

Fields:
   domain -- key into DOMAIN_AGENTS.
   tool_name -- EXACT tool name as wired in the n8n agent's toolset. Passed verbatim into the instruction sent to that agent 
                ("Use ONLY the '...' tool"), so it must match the n8n tool's display name exactly.
   
   description -- one line, what this action does. Shown to workspace_draft_llm.
   payload_fields -- list of {"name", "description"} dicts. Documents both WHAT fields a payload needs and WHAT each one means, 
                    not just the bare names -- this is the "enrich the catalog" change: names alone don't tell an LLM what to put in them.
   
   use_when / dont_use_when -- semantic disambiguation between actions that look similar (reply vs send, add-label vs mark-read, 
                                update vs create). This is what turnsbthe catalog from a name list into actual guidance.
   
   prerequisites (optional) -- freeform list of strings describing state this action depends on (e.g. "needs a thread_id from a 
                                prior read"). Deliberately NOT a fixed boolean flag like `requires_existing_thread` -- different actions have different, sometimes multiple, preconditions, and a free-text list generalizes across all of them without a new schema field per action.
   
   read_only -- True for tools that only fetch data. read_only actions never need human-in-the-loop confirmation.
   requires_confirmation (optional) -- explicit override, defaults to `not read_only`. Its own field (not hardwired 1:1 to read_only) 
                                        so a low-stakes write can later be marked safe to auto-execute by flipping one value here.
"""


TOOL_CATALOG = {
    # ---------- Gmail ----------
    "gmail_get_emails": {
        "domain": "gmail", "tool_name": "Get Emails",
        "description": "Fetch emails from the mailbox matching a search query. Does not modify anything.",
        "payload_fields": [],
        "use_when": "You need to find a message, its thread_id, sender, or body before acting on it (e.g. before replying or marking it read).",
        "dont_use_when": "You already have the thread_id/message_id you need from a prior step in this task.",
        "read_only": True,
    },
    "gmail_get_labels": {
        "domain": "gmail", "tool_name": "Get Labels",
        "description": "List the labels that exist in the mailbox.",
        "payload_fields": [],
        "use_when": "You need a label's exact name before applying it to a message.",
        "dont_use_when": "You already know the exact label name to apply.",
        "read_only": True,
    },
    "gmail_add_labels": {
        "domain": "gmail", "tool_name": "Add Labels",
        "description": "Attach one or more existing labels to a specific message.",
        "payload_fields": [
            {"name": "message_id", "description": "The message to label."},
            {"name": "labels", "description": "List of exact label names to attach."},
        ],
        "use_when": "The user wants an email categorized, flagged, or filed away -- not a reply sent.",
        "dont_use_when": "The user actually wants a reply or new email sent -- this only tags a message, it doesn't communicate anything to anyone.",
        "prerequisites": ["Requires message_id, normally obtained via gmail_get_emails."],
        "read_only": False,
    },
    "gmail_reply_in_thread": {
        "domain": "gmail", "tool_name": "Reply in Thread",
        "description": "Reply to an existing Gmail thread, continuing the conversation.",
        "payload_fields": [
            {"name": "thread_id", "description": "The thread to reply in."},
            {"name": "to", "description": "Recipient address."},
            {"name": "body", "description": "The reply body text."},
        ],
        "use_when": "The user wants to continue an existing email conversation.",
        "dont_use_when": "The user wants a brand new, unrelated email -- use gmail_send_email instead.",
        "prerequisites": ["Requires an existing thread_id, normally obtained via gmail_get_emails."],
        "read_only": False,
    },
    "gmail_draft_email": {
        "domain": "gmail", "tool_name": "Draft Email",
        "description": "Create a draft email without sending it.",
        "payload_fields": [
            {"name": "to", "description": "Recipient address."},
            {"name": "subject", "description": "Email subject line."},
            {"name": "body", "description": "Email body text."},
        ],
        "use_when": "The user explicitly wants a draft to review or send themselves later, not an immediate send.",
        "dont_use_when": "The user wants the email sent now -- use gmail_send_email instead.",
        "read_only": False,
    },
    "gmail_mark_read": {
        "domain": "gmail", "tool_name": "Mark Read",
        "description": "Mark a specific message as read.",
        "payload_fields": [{"name": "message_id", "description": "The message to mark as read."}],
        "use_when": "The user wants inbox cleanup/triage, not a reply.",
        "dont_use_when": "The user wants to respond to the email's content -- marking it read alone doesn't communicate anything.",
        "prerequisites": ["Requires message_id, normally obtained via gmail_get_emails."],
        "read_only": False,
    },
    "gmail_send_email": {
        "domain": "gmail", "tool_name": "Send Email",
        "description": "Send a brand new email that is not a reply within an existing thread.",
        "payload_fields": [
            {"name": "to", "description": "Recipient address."},
            {"name": "subject", "description": "Email subject line."},
            {"name": "body", "description": "Email body text."},
        ],
        "use_when": "The user wants to start a new conversation or send a standalone message.",
        "dont_use_when": "The user is replying within an existing thread -- use gmail_reply_in_thread instead so the conversation stays linked.",
        "read_only": False,
    },

    # ---------- Calendar ----------
    "calendar_create_event": {
        "domain": "calendar", "tool_name": "Create Event",
        "description": "Create a new calendar event.",
        "payload_fields": [
            {"name": "title", "description": "Event title."},
            {"name": "start", "description": "ISO 8601 start datetime with timezone offset."},
            {"name": "end", "description": "ISO 8601 end datetime with timezone offset."},
            {"name": "attendees", "description": "List of attendee email addresses."},
            {"name": "location", "description": "Event location, if any."},
        ],
        "use_when": "The user wants a new meeting/event scheduled that doesn't already exist.",
        "dont_use_when": "The user wants to change an existing event's time/attendees -- use calendar_update_event instead.",
        "prerequisites": ["Consider calendar_check_availability first if the relevant free time isn't already confirmed."],
        "read_only": False,
    },
    "calendar_update_event": {
        "domain": "calendar", "tool_name": "Update Event",
        "description": "Modify fields of an existing calendar event.",
        "payload_fields": [
            {"name": "event_id", "description": "The event to update."},
            {"name": "title", "description": "New title, if changing."},
            {"name": "start", "description": "New ISO 8601 start datetime, if changing."},
            {"name": "end", "description": "New ISO 8601 end datetime, if changing."},
            {"name": "attendees", "description": "New attendee list, if changing."},
            {"name": "location", "description": "New location, if changing."},
        ],
        "use_when": "The user wants to modify an event that already exists.",
        "dont_use_when": "No existing event_id is known -- use calendar_get_events first, or calendar_create_event if this is actually a new event.",
        "prerequisites": ["Requires event_id, normally obtained via calendar_get_events."],
        "read_only": False,
    },
    "calendar_check_availability": {
        "domain": "calendar", "tool_name": "Check Availability",
        "description": "Check whether a given time window is free.",
        "payload_fields": [
            {"name": "start", "description": "ISO 8601 window start."},
            {"name": "end", "description": "ISO 8601 window end."},
        ],
        "use_when": "You need to confirm a time slot is free before proposing or creating an event.",
        "dont_use_when": "The time slot is already confirmed or explicitly fixed by the user.",
        "read_only": True,
    },
    "calendar_get_events": {
        "domain": "calendar", "tool_name": "Get Events",
        "description": "Fetch existing events in a time window.",
        "payload_fields": [
            {"name": "start", "description": "ISO 8601 window start."},
            {"name": "end", "description": "ISO 8601 window end."},
        ],
        "use_when": "You need to find an existing event (and its event_id) before updating/deleting it, or to answer 'what's on my calendar'.",
        "dont_use_when": "You already have the event_id you need.",
        "read_only": True,
    },
    "calendar_delete_event": {
        "domain": "calendar", "tool_name": "Delete Event",
        "description": "Delete an existing calendar event.",
        "payload_fields": [{"name": "event_id", "description": "The event to delete."}],
        "use_when": "The user explicitly wants an existing event removed or cancelled.",
        "dont_use_when": "The user wants to reschedule -- use calendar_update_event instead of delete-then-create.",
        "prerequisites": ["Requires event_id, normally obtained via calendar_get_events."],
        "read_only": False,
    }
}