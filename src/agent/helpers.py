import json
from typing import Optional
from typing_extensions import Literal
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from src.agent.schema import TaskResult

RESET = "__RESET__"

def reduce_list(left: list, right) -> list:
    if right == RESET:
        return []
    if left is None:
        left = []
    if right is None:
        return left
    return left + right

def update_dict(left: dict, right) -> dict:
    """
    Merge reducer for dict-valued fields. Three modes:
    - right is RESET -> field clears to {}.
    - right is {"__replace__": True, "entries": {...}} -> field is replaced wholesale
      (needed to REMOVE keys, which a purely additive merge never can).
    - otherwise -> additive merge {**left, **right}, safe under parallel worker fan-out
      since different workers write different keys.
    """
    if left is None:
        left = {}
    if right is None:
        return left
    if right == RESET:
        return {}
    if isinstance(right, dict) and right.get("__replace__"):
        return right.get("entries", {})
    return {**left, **right}


def increment_counter(left: int, right) -> int:
    """
    Reducer that sums integers for iteration counting, with RESET support so it can
    be zeroed at the start of a new task turn instead of accumulating across the whole
    conversation.
    """
    if right == RESET:
        return 0
    if left is None:
        left = 0
    if right is None:
        return left
    return left + right


def build_indexed_turns(messages: list[BaseMessage], max_turns: int = 8):
    """
    Pairs up Human/AI messages into sequential turn numbers (1, 2, 3...) instead of
    relying on langgraph's internal message UUIDs (fragile for an LLM to reproduce).

    Returns (formatted_string_for_prompt, turns_list). `turns_list` is the SAME
    ordering/window every time this is called against the same `messages` list, which
    is what lets get_referenced_context() resolve indices chosen earlier in the turn.
    Only call this at points where no new Human/AI pair has been appended yet since the
    numbering was shown to the LLM (i.e. anywhere mid-task, before answer_node appends
    the final AIMessage).
    """
    turns = []
    i = 0
    n = len(messages)-1
    while i < n:
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            query = msg.content
            answer = None
            if i + 1 < n and isinstance(messages[i + 1], AIMessage):
                answer = messages[i + 1].content
                i += 2
            else:
                i += 1
            turns.append({"query": query, "answer": answer})
        else:
            i += 1

    turns = turns[-max_turns:]

    if not turns:
        return "No prior conversation.", turns

    lines = [
        f"[{idx}] User: {t['query']}\n    Assistant: {t['answer'] or '(no answer recorded)'}"
        for idx, t in enumerate(turns, start=1)
    ]
    return "\n".join(lines), turns


def get_referenced_context(state, referenced_ids: Optional[list[int]]) -> str:
    """
    Resolves turn numbers picked by assistant_node/supervisor_node back into actual
    message content, for workspace_agent/answer_node to use.
    """
    if not referenced_ids:
        return ""

    _, turns = build_indexed_turns(state.get("messages", []))
    parts = []
    for rid in referenced_ids:
        if 1 <= rid <= len(turns):
            t = turns[rid - 1]
            parts.append(f"[Referenced Turn {rid}]\nUser: {t['query']}\nAssistant: {t['answer'] or '(no recorded answer)'}")
    return "\n\n".join(parts)


def build_supervisor_context(state) -> dict:
    """
    Lightweight signals for the supervisor -- cheap indices/flags, never full content or history.
    """
    kb_meta = state.get("external_kb_meta") or {}
    uploaded_doc_summary = state.get("uploaded_doc_summary")

    if kb_meta.get("available"):
        kb_context = f"""
        An INTERNAL knowledge base is available. Topic: '{kb_meta.get('topic', 'unknown')}'. Summary: {kb_meta.get('summary', 'N/A')}.
        This is separate from any uploaded documents (see Notebook). Instruct research_agent to use 'internal_kb_search' when relevant.
        """
    else:
        kb_context = "No internal knowledge base is available."

    if uploaded_doc_summary is not None:
        doc_context = f"User has uploaded a document. Summary of the doc: {uploaded_doc_summary}"
    else:
        doc_context = "No user uploaded document available."

    return {
        "kb_context": kb_context,
        "doc_context": doc_context,
        "reference_note": state.get("reference_note") or "No referenced context for this request.",
    }


def deduplicate_task_results(task_results: list[TaskResult]) -> list[TaskResult]:
    """Remove exact duplicate TaskResult entries while preserving order. Dicts aren't hashable, so
    dedupe on a JSON-serialized key instead of a plain set-of-strings."""
    seen = set()
    deduped = []
    for result in task_results:
        key = json.dumps(result, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(result)
    return deduped


def task_result(agent: str, task: str, status: Literal["success", "error"], output: str, metadata: Optional[dict] = None) -> TaskResult:
    return {
        "agent": agent,
        "task": task,
        "status": status,
        "output": output,
        "metadata": metadata or {}
        }                                  # type: ignore                                     


def format_task_result(tr: TaskResult) -> str:
    """The one place a TaskResult gets turned into text, for prompts that need a string."""
    marker = "OK" if tr.get("status") == "success" else "ERROR"
    meta = tr.get("metadata") or {}
    meta_str = f" | {meta}" if meta else ""

    return f"[{tr.get('agent')}] task={tr.get('task')!r} status={marker}: {tr.get('output')}{meta_str}"


def needs_confirmation(entry: dict) -> bool:
    return entry.get("requires_confirmation", not entry["read_only"])


def format_catalog_entry(key: str, entry: dict) -> str:
    """
    Renders one TOOL_CATALOG entry into the semantic, LLM-readable block (name / description /
    requires / use when / don't use when / prerequisites / confirmation) instead of just a bare
    tool name. Generalized over whatever optional fields a given entry actually has -- no field is
    assumed present except domain/tool_name/payload_fields/read_only, so this works unchanged as
    new actions (and new optional fields) get added to the catalog.
    """
    lines = [key, f"  n8n tool: {entry['tool_name']}  (domain: {entry['domain']})"]

    if entry.get("description"):
        lines.append(f"  Description: {entry['description']}")

    payload_fields = entry.get("payload_fields") or []
    if payload_fields:
        rendered = "; ".join(
            f"{f['name']} ({f['description']})" if isinstance(f, dict) and f.get("description") else
            (f["name"] if isinstance(f, dict) else str(f))
            for f in payload_fields
        )
        lines.append(f"  Requires payload: {rendered}")
    else:
        lines.append("  Requires payload: none")

    if entry.get("use_when"):
        lines.append(f"  Use when: {entry['use_when']}")
    if entry.get("dont_use_when"):
        lines.append(f"  Don't use when: {entry['dont_use_when']}")

    prereqs = entry.get("prerequisites")
    if prereqs:
        prereqs = prereqs if isinstance(prereqs, list) else [prereqs]
        lines.append(f"  Prerequisites: {'; '.join(prereqs)}")

    lines.append(
        f"  Confirmation: {'required -- human must approve before this runs' if needs_confirmation(entry) else 'not required -- executes immediately'}"
    )
    return "\n".join(lines)