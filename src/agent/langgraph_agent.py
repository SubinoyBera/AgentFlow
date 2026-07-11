import os, sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional
from typing_extensions import TypedDict, Literal, Annotated
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sentence_transformers import CrossEncoder
from pathlib import Path
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from src.logger.logging import logging
from src.exception.exception_handler import AppException
from utils.common import get_checkpointer
from src.tools.research_tools import retriever, tavily_search, news_search, wiki_search, weather_tool, stock_finance_tool
from src.tools.workspace_automation import gmail_read_tool, gmail_send_tool, calendar_read_tool, calendar_write_tool
from src.agent.prompt import (assistant_system_prompt, supervisor_system_prompt, research_agent_system_prompt,
                    workspace_agent_system_prompt, answer_agent_system_prompt, vision_agent_system_prompt, document_agent_system_prompt)


# CONSTANTS
MAX_SUPERVISOR_ITERATIONS = 10
MAX_REFERENCE_TURNS = 10
WORKSPACE_TIMEZONE = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# STATE REDUCERS
# ---------------------------------------------------------------------------

class _Reset:
    """
    Sentinel: returning this as a field's value tells its reducer to clear the field
    back to empty, instead of merging/appending.
    """
    pass

RESET = _Reset()

def reduce_list(left: list, right) -> list:
    if right is RESET:
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
    if right is RESET:
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
    if right is RESET:
        return 0
    if left is None:
        left = 0
    if right is None:
        return left
    return left + right


# AGENTS STATE SCHEMA
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    external_kb_meta: dict
    plan: list[str]
    completed_steps: list[str]
    task_results: Annotated[list[str], reduce_list]
    delegation_instructions: Annotated[dict[str, str], update_dict]
    next_nodes: list[str]
    uploaded_doc: Optional[str]
    uploaded_doc_summary: Optional[str]
    uploaded_image: bool
    image_data: Optional[list]
    doc_context_mode: Optional[str]  
    reference_context_ids: Optional[list[int]]
    pending_action: Optional[dict]
    hitl_feedback: Optional[str]
    iteration_count: Annotated[int, increment_counter]


# OUTPUT SCHEMA FOR GENERAL ASSISTANT
class AssistantDecision(BaseModel):
    decision: Literal["direct_answer", "ambiguous", "task", "vision"] = Field(
        description=(
            """
            direct_answer: answerable directly and confidently from general knowledge and available context without any research or tools. e.g. greetings/ small talk, summarization, etc.
            ambiguous: genuine fork in meaning that changes the correct answer and cannot be safely guessed without any relevant context (e.g. 'cricket' = sport or insect).
            task: requires research, using external tools, accessing uploaded documents, accessing internal Knowlege Base, etc.
            vision: if image is uploaded, always route to 'vision'.
            """
        )
    )
    response: Optional[str] = Field(
        default=None,
        description="Required for 'direct_answer' and 'ambiguous' (clarifying question). Should be None for 'task' and 'vision'."
    )
    reframed_query: Optional[str] = Field(
        default=None,
        description=(
            """
            Required for 'task' and 'vision' only. Rewrite the user's latest query into a fully self-contained query, resolving pronouns/references via Past Conversations
            (e.g. 'explain it in more depth' -> 'explain [topic from history] in more depth'). If already self-contained, return unchanged.
            """
        )
    )
    reference_context_ids: Optional[list[int]] = Field(
        default=None,
        description=(
            "If the query refers back to earlier turns (e.g. 'reply to that email', 'expand on that'), "
            "list the turn numbers shown in the numbered Past Conversations block that are relevant. "
            "Omit/None if nothing needs to be referenced."
        )
    )


# OUTPUT SCHEMA FOR THE SUPERVISOR ORCHESTRATOR
class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Reasoning for the next step")
    plan: list[str] = Field(description="Step-by-step plan of sub-tasks to execute. Keep the existing plan or update/modify it if plan of action changes (e.g. if a tool fails).")
    completed_steps: list[str] = Field(description="List of tasks from the plan that have been completed so far. Add newly completed tasks to this list.")
    next_nodes: list[str] = Field(description="Agents to route to next. E.g., ['research_agent'] or ['research_agent', 'workspace_agent']. Use ['FINISH'] to end.")
    delegation_instructions: dict[str, str] = Field(description="Specific instructions for each agent selected in next_nodes. Key is the agent name.", default_factory=dict)
    doc_context_mode: Literal["none", "document"] = Field(
        default="document",
        description="""Determines what context the document agent receives. 'document' = only uploaded doc"""
    )


class WorkspaceDraftDecision(BaseModel):
    """
    Used by the READ-ONLY drafting stage of workspace_agent. Bound to structured output on purpose
    instead of letting a tool-calling agent decide for itself -- that's what makes it physically
    impossible for this stage to trigger a send/create/delete action; it can only read and propose.
 
    Generic on purpose: action_type/payload/confirmation_preview work for any registered action
    (see ACTION_REGISTRY below) without this schema needing a new field every time a new action
    type (Slack, file delete, ...) gets added.
    """
    action_type: str = Field(
        description=(
            "'read_only' if the request was fully answered by reading (no confirmation needed), "
            "otherwise the exact key of the action in ACTION_REGISTRY that this should perform "
            "(e.g. 'gmail_send', 'calendar_create', 'calendar_update', 'calendar_delete'). "
            "Never perform the action yourself -- only draft it."
        )
    )
    direct_result: Optional[str] = Field(
        default=None,
        description="Required for 'read_only'. The complete answer to show the user."
    )
    payload: Optional[dict] = Field(
        default=None,
        description=(
            "Required for anything other than 'read_only'. The exact fields the execute step for "
            "this action_type needs -- e.g. for 'gmail_send': {to, subject, body, thread_id}; for "
            "'calendar_create': {title, start, end, attendees, location}."
        )
    )
    confirmation_preview: Optional[str] = Field(
        default=None,
        description=(
            "Required for anything other than 'read_only'. A short, human-readable preview of the "
            "drafted action for the user to approve/reject -- e.g. 'Reply to Jane Doe: \"Sounds "
            "good, see you Tuesday.\"' or 'New event: Team Sync, Tue 3-4pm, with Alex and Priya.'"
        )
    )

# OUTPUT SCHEMA FOR ANSWER AGENT
class AnswerAgentResponse(BaseModel):
    final_answer: Optional[str] = Field(default=None, description="The final answer generated by the answer agent. Set to None if more information is needed.")


# INITIALIZE LLMs
assistant_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5).with_structured_output(AssistantDecision)
supervisor_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2).with_structured_output(SupervisorDecision)
research_agent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.4)
workspace_agent_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)
workspace_draft_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3).with_structured_output(WorkspaceDraftDecision)
answer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6).with_structured_output(AnswerAgentResponse)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
vision_llm = genai.GenerativeModel("gemini-2.5-flash")

# load reranker model for retriever tool
reranker_path = Path("models/bge-reranker-base")
reranker = CrossEncoder(str(reranker_path))


def _execute_gmail_send(payload: dict) -> str:
    return gmail_send_tool.invoke({
        "to": payload.get("to"),
        "subject": payload.get("subject"),
        "body": payload.get("body"),
        "thread_id": payload.get("thread_id"),
    })
 
def _execute_calendar_create(payload: dict) -> str:
    return calendar_write_tool.invoke({"action": "create", "event_details": payload})
 
def _execute_calendar_update(payload: dict) -> str:
    return calendar_write_tool.invoke({"action": "update", "event_details": payload})
 
def _execute_calendar_delete(payload: dict) -> str:
    return calendar_write_tool.invoke({"action": "delete", "event_details": payload})
 
 
ACTION_REGISTRY = {
    "gmail_send": {
        "read_tool": gmail_read_tool,
        "execute": _execute_gmail_send,
        "payload_fields": ["to", "subject", "body", "thread_id"],
    },
    "calendar_create": {
        "read_tool": calendar_read_tool,
        "execute": _execute_calendar_create,
        "payload_fields": ["title", "start", "end", "attendees", "location"],
    },
    "calendar_update": {
        "read_tool": calendar_read_tool,
        "execute": _execute_calendar_update,
        "payload_fields": ["event_id", "title", "start", "end", "attendees", "location"],
    },
    "calendar_delete": {
        "read_tool": calendar_read_tool,
        "execute": _execute_calendar_delete,
        "payload_fields": ["event_id"],
    }
}
 
_REGISTRY_TOOLS_DOC = "\n".join(
    f"- {name}: needs payload fields {spec['payload_fields']}" for name, spec in ACTION_REGISTRY.items()
)
 


# GENERAL HELPERS
def deduplicate_task_results(task_results: list[str]) -> list[str]:
    """Remove exact duplicate entries from task_results while preserving order."""
    seen = set()
    deduped = []
    for result in task_results:
        if result not in seen:
            seen.add(result)
            deduped.append(result)
    
    return deduped


def build_indexed_chat_turns(messages: list[BaseMessage], max_turns: int = MAX_REFERENCE_TURNS):
    """
    Pairs up Human/AI messages into sequential turn numbers (1, 2, 3...) instead of relying on langgraph's internal message UUIDs (fragile for an LLM to reproduce).
 
    Returns (formatted_string_for_prompt, turns_list). `turns_list` is the SAME ordering/window every time this is called against the same `messages` list, which
    is what lets get_referenced_context() resolve indices chosen earlier in the turn.
    """
    turns = []
    i = 0
    n = len(messages)
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
        f"[{idx}] User: {t['query']} \nAssistant: {t['answer'] or '(no answer recorded)'}"
        for idx, t in enumerate(turns, start=1)
    ]
    return "\n".join(lines), turns
 
 
def get_referenced_context(state: "AgentState", referenced_ids: Optional[list[int]]) -> str:
    """
    Resolves turn numbers picked by assistant_node/supervisor_node back into actual
    message content, for workspace_agent/answer_node to use.
    """
    if not referenced_ids:
        return ""
 
    _, turns = build_indexed_chat_turns(state.get("messages", []))
    parts = []
    for rid in referenced_ids:
        if 1 <= rid <= len(turns):
            t = turns[rid - 1]
            parts.append(f"[Referenced Turn {rid}] \nUser: {t['query']} \nAssistant: {t['answer'] or '(no recorded answer)'}")
    
    return "\n\n".join(parts)
 
 

def build_supervisor_context(state: AgentState) -> dict:
    """
    Lightweight signals for the supervisor — cheap indices/flags, never full content or history.
    """
    kb_meta = state.get("external_kb_meta", {})
    uploaded_doc_summary = state.get("uploaded_doc_summary", None)

    if kb_meta.get("available"):
        kb_context = f"""
        An INTERNAL knowledge base is available. Topic: '{kb_meta.get('topic', 'unknown')}'. Summary: {kb_meta.get('summary', 'N/A')}. 
        This is separate from any uploaded documents (see Notebook). Instruct research_agent to use 'internal_kb_search' when relevant.
        """
    else:
        kb_context = "No internal knowledge base is available."

    if uploaded_doc_summary is not None:
        doc_context = f"""
        User has uploaded a document. Summary of the doc: {uploaded_doc_summary}
        """
    else:
        doc_context = "No user uploaded document available."

    return {
        "kb_context": kb_context,
        "doc_context": doc_context,
        "reference_index": state.get("reference_context_ids", None)
    }


# NODE 0: GENERAL ASSISTANT
def assistant_node(state: AgentState):
    """
    Context-aware front door assistant. Sees chat history (nothing downstream of this node does). 
    Handles chitchat messages, ambiguity, follow-ups directly, and reframes task queries into self-contained form before handing off to the supervisor.
    """
    query = state.get("query", "User query not found.")
    is_doc_uploaded = True if state.get("uploaded_doc", "None") is not None else False
    is_image_uploaded = state.get("uploaded_image", None)
    pending_action = state.get("pending_action", None)
    hitl_feedback = state.get("hitl_feedback", None)
    indexed_chat_history, _ = build_indexed_chat_turns(state.get("messages", []))

    try:
        chat_template = ChatPromptTemplate(
            [
                ("system", assistant_system_prompt),
                ("human", 
                    """Previous Conversations (numbered turns -- use these numbers if you need to reference one): \n{chat_history}\n\n
                    Current user query: {query}\n\n Document uploaded: {is_doc_uploaded} \nImage uploaded: {is_image_uploaded}
                    """
                )
            ],
            input_variables=["chat_history", "query", "is_doc_uploaded", "is_image_uploaded"]
        )
        chain = chat_template | assistant_llm
        response: AssistantDecision = chain.invoke({
            "chat_history": indexed_chat_history,
            "query": query,
            "is_doc_uploaded": is_doc_uploaded,
            "is_image_uploaded": is_image_uploaded})

    except Exception as e:
        logging.error(f"Error in assistant node: {e}", exc_info=True)

    if not response or response.decision == "task":
        resolved_query = response.reframed_query if response else query
        ref_ids = response.reference_context_ids if response else None

        return {
            "query": resolved_query,
            "next_nodes": ["supervisor"],
            "plan": [],
            "completed_steps": [],
            "task_results": RESET,
            "delegation_instructions": RESET,
            "reference_context_ids": ref_ids if ref_ids else RESET,
            "pending_action": pending_action,
            "hitl_feedback": hitl_feedback,
            "iteration_count": RESET
        }
    
    if response.decision == "vision":
        return {
            "query": query,
            "next_nodes": ["vision"]
        }

    return {
        "messages": [AIMessage(content=response.response)], 
        "next_nodes": ["FINISH"]
    }


# NODE 1: THE SUPERVISOR ORCHESTRATOR
def supervisor_node(state: AgentState):
    query = state.get("query", "")
    if not query and state.get("messages"):
        query = str(state["messages"][-1].content)

    task_results = deduplicate_task_results(state.get("task_results", []))
    current_plan = state.get("plan", [])
    completed_steps = state.get("completed_steps", [])
    reference_ids = state.get("reference_context_ids")
    iteration_count = state.get("iteration_count", 0)

    task_results_str = "\n".join(task_results) if task_results else "None"

    if iteration_count >= MAX_SUPERVISOR_ITERATIONS:
        logging.warning(f"Supervisor hit max iterations ({MAX_SUPERVISOR_ITERATIONS}) for this task. Force-routing to answer_agent.")
        return {
            "plan": current_plan,
            "completed_steps": completed_steps,
            "next_nodes": ["answer_agent"],
            "delegation_instructions": {"__replace__": True, "entries": {
                "answer_agent": """Maximum processing iterations reached for this request. Please provide the best possible answer using the information gathered so far, 
                and be honest that the process was cut short."""
            }},
            "iteration_count": 1,
            "intermediate_query": None
        }
        
    # Context Compression Logic for task_results (within-task; unrelated to notebook)
    if len(task_results_str) > 3000:
        logging.info(f"Task results string length ({len(task_results_str)}) exceeds 3000 chars. Compressing...")
        try:
            summary_prompt = f"Summarize the following task results concisely, keeping all critical facts, statuses, and agent outputs necessary for a supervisor to plan the next steps:\n\n{task_results_str}"
            summary_response = research_agent_llm.invoke(summary_prompt)
            task_results_str = f"[COMPRESSED SUMMARY of previous steps]:\n{summary_response.content}"
        except Exception as e:
            logging.error(f"Context compression failed: {e}")

    # Build supervisor context
    context = build_supervisor_context(state)

    try:
        chat_template = ChatPromptTemplate(
            [
                ("system", supervisor_system_prompt),
                ("human",
                 """Original Question: {query}\n\n
                 Internal KB: {kb_context} \n\nUser uploaded document summary: {doc_context}\n\n
                 Referenced past chat turns (indices): {reference_index}\n\n
                 Current Plan: \n{plan} \n\nCompleted Steps: \n{completed_steps}\n\n
                 Task Results so far: \n{task_results}"""
                )
            ],
            input_variables=["query", "kb_context", "doc_context", "notebook_index",
                            "plan", "completed_steps", "task_results"]
        )

        chain = chat_template | supervisor_llm
        response: SupervisorDecision = chain.invoke({
            "query": query,
            "plan": "\n".join(f"- {step}" for step in current_plan) if current_plan else "None",
            "completed_steps": "\n".join(f"- {step}" for step in completed_steps) if completed_steps else "None",
            "task_results": task_results_str,
            **context,
        })

        return {
            "plan": response.plan,
            "completed_steps": response.completed_steps,
            "next_nodes": response.next_nodes,
            "delegation_instructions": {"__replace__": True, "entries": response.delegation_instructions},
            "doc_context_mode": response.doc_context_mode,
            "reference_context_ids": response.reference_index if response.reference_index else state.get("reference_context_ids"),
            "iteration_count": 1
        }
    except Exception as e:
        logging.error(f"Error in Supervisor node: {e}", exc_info=True)
        raise AppException(e, sys)


# NODE 2: WORKSPACE AGENT NODE
def workspace_agent(state: AgentState):
    """
    The mini-supervisor for workspace actions. The global supervisor only ever says "handle this with workspace_agent" -- it has no idea whether that means Gmail, Calendar, etc..
    This node decides all of that:
 
    1. Gathers context using every READ-ONLY tool in ACTION_REGISTRY (safe to freely tool-call -- none of them can write anything).
    2. Picks an action_type from ACTION_REGISTRY (or 'read_only') via structured output -- not tool-calling -- so it is physically incapable of triggering a send/create/delete itself.
    3. Read-only requests finish here. Anything else becomes a pending_action and routes to workspace_confirm for human approval; the actual write only ever happens in workspace_send.
 
    Adding a new capability (Slack, file delete, ...) means adding read/execute tools in workspace_automation.py and one ACTION_REGISTRY entry -- nothing in this function changes.
    """
    instruction = state.get("delegation_instructions", {}).get("workspace_agent", "No instruction provided")
    referenced_context = get_referenced_context(state, state.get("reference_context_ids"))
    feedback = state.get("hitl_feedback")
    previous = state.get("pending_action") or {}
 
    # Anchor for resolving relative dates 
    now_str = datetime.now(WORKSPACE_TIMEZONE).strftime("%A, %B %d, %Y %I:%M %p %Z")
 
    full_instruction = f"Current date/time: {now_str}\n\n"
    full_instruction += f"{referenced_context}\n\n---\n\nCurrent request: {instruction}" if referenced_context else f"Current request: {instruction}"
    if feedback:
        full_instruction += (
            f"\n\n---\n\nYou previously drafted this ({previous.get('type')}): \n{previous.get('preview')}\n\n"
            f"The user asked for changes: {feedback} \nGather any additional context needed and revise."
        )
 
    # 1: Bind every registered read tool -- the model picks whichever domain(s) are relevant; none of them can write, so this is safe as freeform tool-calling.
    read_tools = list({spec["read_tool"].name: spec["read_tool"] for spec in ACTION_REGISTRY.values()}.values())
    reader = create_agent(
        model=workspace_agent_llm,
        tools=read_tools,
        system_prompt=workspace_agent_system_prompt,
    )
    config = RunnableConfig({"recursion_limit": 4})
 
    try:
        reader_response = reader.invoke({"messages": [{"role": "user", "content": full_instruction}]}, config=config)
        gathered_context = reader_response["messages"][-1].content
    
    except Exception as e:
        logging.error(f"Error gathering workspace context: {e}", exc_info=True)
        return {
            "task_results": [f"[workspace_agent] Error gathering context: {str(e)}"],
            "pending_action": None
        }
 
    # 2: Structured decision -- read_only (done) vs a drafted action_type from ACTION_REGISTRY.
    try:
        draft_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You turn gathered workspace context into either a direct answer (read_only requests) or a drafted action awaiting human approval. Never claim an action was completed -- 
             you can only draft it, never perform it. When a payload needs a date/time (e.g. calendar start/end), resolve it against the Current date/time given in the request and output 
             it in ISO 8601 format with timezone offset (e.g. 2026-07-14T15:00:00+05:30).\n\n Available actions:\n{registry_doc}"""),
            ("human", "Request: {instruction} \n\nGathered context: \n{gathered_context}"),
        ])

        decision: WorkspaceDraftDecision = (draft_prompt | workspace_draft_llm).invoke({
            "instruction": instruction,
            "gathered_context": gathered_context,
            "registry_doc": _REGISTRY_TOOLS_DOC,
        })

    except Exception as e:
        logging.error(f"Error producing workspace draft decision: {e}", exc_info=True)
        return {"task_results": [f"[workspace_agent] Error drafting action: {str(e)}"], "pending_action": None}
 
    if decision.action_type == "read_only":
        return {
            "task_results": [f"[workspace_agent result for '{instruction}']: {decision.direct_result or gathered_context}"],
            "pending_action": None,
        }
 
    if decision.action_type not in ACTION_REGISTRY:
        logging.error(f"workspace_agent produced an unregistered action_type: {decision.action_type}")
        return {
            "task_results": [f"[workspace_agent] Couldn't determine how to safely perform this action ('{decision.action_type}')."],
            "pending_action": None,
        }
 
    # Side-effecting action -- stage it, do NOT execute. route_from_workspace_draft sends this to workspace_confirm next.
    return {
        "pending_action": {
            "type": decision.action_type,
            "payload": decision.payload or {},
            "preview": decision.confirmation_preview or "No preview available.",
        },
        "hitl_feedback": None,
    }

def workspace_confirm(state: AgentState):
    """
    Pauses the graph and hands the drafted action to the caller for approval.
    `interrupt()` halts execution here (LangGraph checkpoints state) until the calling code resumes with: ai_agent.invoke(Command(resume=decision), config)
    where `decision` is a dict like {"approved": True} or {"approved": False, "feedback": "make it shorter"}. 
 
    Note: everything in a node BEFORE its interrupt() call re-runs on resume (LangGraph replays the node from the top). This node does nothing but read `pending_action` 
    and interrupt -- keep it that way, don't add tool calls here.
    """
    pending = state.get("pending_action") or {}
    decision = interrupt({
        "type": "workspace_action_confirmation",
        "action_type": pending.get("type"),
        "preview": pending.get("preview"),
        "payload": pending.get("payload"),
    })
 
    if decision.get("approved"):
        return {"hitl_feedback": None}
    return {
        "hitl_feedback": decision.get("feedback")
        or "The user rejected the draft without specific feedback -- ask what they'd like changed."
    }
 
 
def workspace_send(state: AgentState):
    """
    Runs ONLY after workspace_confirm records approval. Looks up the approved action_type in
    ACTION_REGISTRY and calls its execute function with the approved payload -- this is the only
    place a write ever actually happens. Adding a new action type never touches this function.
    """
    pending = state.get("pending_action") or {}
    action_type = pending.get("type")
    entry = ACTION_REGISTRY.get(action_type)
 
    if not entry:
        logging.error(f"workspace_send got an unregistered action_type: {action_type}")
        return {
            "task_results": [f"[workspace_agent] Unknown action type '{action_type}' -- nothing was executed."],
            "pending_action": None,
            "hitl_feedback": None,
        }
 
    try:
        result = entry["execute"](pending.get("payload") or {})
        return {
            "task_results": [f"[workspace_agent] {action_type} completed: {result}"],
            "pending_action": None,
            "hitl_feedback": None,
        }
    except Exception as e:
        logging.error(f"Error executing approved workspace action ({action_type}): {e}", exc_info=True)
        return {
            "task_results": [f"[workspace_agent] Error executing {action_type}: {str(e)}"],
            "pending_action": None,
            "hitl_feedback": None,
        }

# NODE 3: RESEARCH NODE
def research_node(state: AgentState):
    instruction = state.get("delegation_instructions", {}).get("research_agent", state.get("query", ""))

    @tool
    def internal_kb_search(query: str):
        """Search the INTERNAL business knowledge base (RAG) for relevant documents.
        This is NOT for user-uploaded PDFs — those are handled by document_agent using
        the full document text directly, never RAG."""
        index_name = state.get("external_kb_meta", {}).get("topic", "")
        if not index_name:
            return "No internal knowledge base is available. Use web search tools instead."
        docs = retriever.invoke({"query": query, "index_name": index_name, "reranker": reranker})
        return docs if docs else "No relevant documents found in internal knowledge base."

    tools = [internal_kb_search, tavily_search, news_search, wiki_search, weather_tool, stock_finance_tool]

    kb_meta = state.get("external_kb_meta", {})
    if kb_meta.get("available"):
        research_agent_system_prompt += f"\n\nNOTE: An internal knowledge base is available on topic '{kb_meta.get('topic', 'unknown')}'. If the instruction relates to this topic, use the `internal_kb_search` tool."

    research_agent = create_agent(
        model=research_agent_llm,
        tools=tools,
        system_prompt=research_agent_system_prompt
    )
    config = RunnableConfig({"recursion_limit": 6})

    try:
        response = research_agent.invoke({"messages": [{"role": "user", "content": instruction}]}, config=config)
        result = response['messages'][-1].content
        
        return {
            "task_results": [f"[research_agent result for '{instruction}']: {result}"]
        }
    
    except Exception as e:
        logging.error(f"Error in research node: {e}", exc_info=True)
        return {"task_results": [f"[research_agent] Error during research: {str(e)}"]}


# NODE 4: ANSWER NODE 
def answer_node(state: AgentState):
    instruction = state.get("delegation_instructions", {}).get("answer_agent", state.get("query", ""))
    doc_context_mode = state.get("doc_context_mode")
    referenced_context = get_referenced_context(state, state.get("reference_context_ids"))

    try:
        task_results = deduplicate_task_results(state.get("task_results", []))

        if doc_context_mode == "document":
            uploaded_doc = state.get("uploaded_doc")
            web_results = [r for r in task_results if r.startswith("[research_agent result")]

            prompt = "Uploaded Document: \n{uploaded_doc} \n\nWeb Search Results: \n{web_results} \n\nLast Conversation: \n{referenced_context} \n\nQuestion: \n{instruction}"
            variables = {
                "uploaded_doc": uploaded_doc,
                "web_results": "\n".join(web_results) if web_results else "None",
                "referenced_context": referenced_context,
                "instruction": instruction,
            }
        else:
            prompt = "Available Context:\n{task_results} \n\nReferenced Past Chat turns:\n{referenced_context} \n\nQuestion:{instruction}"
            variables = {
                "task_results": "\n".join(task_results) if task_results else "None",
                "referenced_context": referenced_context,
                "instruction": instruction,
            }

        chat_template = ChatPromptTemplate(
        [
            ("system", answer_agent_system_prompt),
            ("human", prompt)
        ], input_variables=[]
        )
        
        chain = chat_template | answer_llm

        response: AnswerAgentResponse = chain.invoke(variables)
        final_answer = response.final_answer or "I apologize, but I was unable to generate a proper answer."
        
        return {
            "messages": [AIMessage(content=final_answer)],
            "next_nodes": ["FINISH"]
        }
 
    except Exception as e:
        logging.error(f"Error in answer node: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I apologize, but I encountered an error generating the final answer. Please try again later.")],
            "next_nodes": ["FINISH"]
        }


# NODE 6 : VISION NODE
def vision_node(state: AgentState):
    try:
        image_data = state.get("image_data", None)
        if image_data is not None:
            response = vision_llm.generate_content([vision_agent_system_prompt, state["query"], image_data[0]])
            return {"response": response.text, "messages": [AIMessage(content=response.text)]}
    
    except Exception as e:
        logging.error(f"Error in multimodal node: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"I apologize, but I encountered an error!! Please try again later.")]}


# ---------------------------------------------------------------------------
# CONDITIONAL ROUTING FUNCTIONS
# ---------------------------------------------------------------------------

def route_from_assistant(state: AgentState) -> str:
    nodes = state.get("next_nodes", ["supervisor"])
    if "FINISH" in nodes:
        return END
    if "vision" in nodes:
        return "vision"
    return "supervisor"
 
def route_from_supervisor(state: AgentState) -> list[str]:
    nodes = state.get("next_nodes", ["FINISH"])
    if "FINISH" in nodes:
        return [END]
    return nodes

def route_from_workspace_draft(state: AgentState) -> str:
    # workspace_agent sets pending_action when it drafted a side-effecting action that needs human approval; a read_only result leaves pending_action None and just loops to supervisor.
    if state.get("pending_action"):
        return "workspace_confirm"
    return "supervisor"

def route_from_confirm(state: AgentState) -> str:
    # hitl_feedback is set (non-None) when the user rejected/asked for edits -> redraft. hitl_feedback is cleared to None on approval -> actually send/create/etc.
    if state.get("hitl_feedback"):
        return "workspace_agent"
    return "workspace_send"


# ---------------------------------------------------------------------------
# BUILD GRAPH WORKFLOW
# ---------------------------------------------------------------------------
graph = StateGraph(AgentState)
 
graph.add_node("assistant", assistant_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("research_agent", research_node)
graph.add_node("workspace_agent", workspace_agent)
graph.add_node("workspace_confirm", workspace_confirm)
graph.add_node("workspace_send", workspace_send)
graph.add_node("answer_agent", answer_node)
graph.add_node("vision", vision_node)
 
graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", route_from_assistant)
graph.add_conditional_edges("supervisor", route_from_supervisor)
 
graph.add_edge("research_agent", "supervisor")
graph.add_conditional_edges("workspace_agent", route_from_workspace_draft)
graph.add_conditional_edges("workspace_confirm", route_from_confirm)
graph.add_edge("workspace_send", "supervisor")
graph.add_edge("vision", END)
graph.add_edge("answer_agent", END)


checkpointer = get_checkpointer()
ai_agent = graph.compile(checkpointer=checkpointer)