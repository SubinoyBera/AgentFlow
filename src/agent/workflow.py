import os, sys
from typing import Optional
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from langchain_core.runnables import RunnableConfig
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from src.logger import logging
from src.exception.exception_handler import AppException
from src.agent.helpers import *
from src.agent.schema import (AssistantDecision, SupervisorDecision, WorkspaceAgentDecission, AnswerAgentResponse, TaskResult)
from src.agent.prompt import (assistant_system_prompt, supervisor_system_prompt, research_agent_system_prompt,
                              answer_agent_system_prompt, vision_agent_system_prompt)


# AGENT STATE SCHEMA
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    internal_kb_meta: dict
    plan: list[str]
    completed_steps: list[str]
    task_results: Annotated[list[TaskResult], reduce_list]
    delegation_instructions: Annotated[dict[str, str], update_dict]
    next_nodes: list[str]
    uploaded_doc: Optional[str]
    uploaded_doc_summary: Optional[str]
    image_data: Optional[list]
    reference_context_ids: Optional[list[int]]
    reference_note: Optional[str]
    doc_context_mode: Optional[str]
    pending_action: Optional[dict]
    draft_answer: Optional[str]
    requires_hitl: bool
    hitl_feedback: Optional[str]
    is_final_step: bool


# INITIALIZE LLMs
assistant_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4).with_structured_output(AssistantDecision)
supervisor_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2).with_structured_output(SupervisorDecision)
research_agent_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
workspace_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1).with_structured_output(WorkspaceAgentDecission)
answer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6).with_structured_output(AnswerAgentResponse)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))            #type: ignore
vision_llm = genai.GenerativeModel("gemini-2.5-flash")          #type: ignore

# INITIALIZE TOOLS
mcp_tools: list = []
n8n_agents = {}


# HANDLE INTERRUPTS
def _parse_resume_decision(decision) -> tuple[bool, Optional[str]]:
    """
    Normalizes whatever comes back from interrupt() into (approved, feedback).
    Standardized interrupt/resume contract. Every HITL pause uses the SAME payload shape going out:
       {"type": "approval", "message": <str>, ...extra context...}
    """
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        feedback = decision.get("feedback")
        return approved, feedback
    
    # Fallback for a raw "y"/"n" string
    #if isinstance(decision, str):
        return decision.strip().lower() in ("y", "yes", "true", "1"), None
    return False, None


# NODE 0: GENERAL ASSISTANT
def assistant_node(state: AgentState):
    """
    Context-aware front door assistant. Sees chat history (nothing downstream of this node does).
    Handles chitchat, ambiguity, follow-ups directly, and reframes task queries into self-contained
    form before handing off to the supervisor. Also picks which past turns (by number) should be
    carried forward for downstream agents.
    """
    query = state.get("query", "User query not found.")
    is_doc_uploaded = bool(state.get("uploaded_doc"))
    is_image_uploaded = bool(state.get("image_data"))

    indexed_history, _ = build_indexed_turns(state.get("messages", []))

    response: Optional[AssistantDecision] = None
    try:
        chat_template = ChatPromptTemplate(
            [
                ("system", assistant_system_prompt),
                ("human",
                 "Previous Conversations (numbered turns -- use these numbers if you need to reference one):\n"
                 "{chat_history}\n\nCurrent user query: {query}\n\n"
                 "Document uploaded: {is_doc_uploaded}\nImage uploaded: {is_image_uploaded}")
            ],
            input_variables=["chat_history", "query", "is_doc_uploaded", "is_image_uploaded"]
        )
        
        chain = chat_template | assistant_llm
        response = chain.invoke({
            "chat_history": indexed_history,
            "query": query,
            "is_doc_uploaded": is_doc_uploaded,
            "is_image_uploaded": is_image_uploaded,
        })                                                    #type: ignore   

    except Exception as e:
        logging.error(f"Error in assistant node: {e}", exc_info=True)
        response = None

    if not response or response.decision == "task":
        resolved_query = response.reframed_query if response and response.reframed_query else query
        ref_ids = response.reference_context_ids if response else None
        return {
            "query": resolved_query,
            "next_nodes": ["supervisor"],
            "plan": [],
            "completed_steps": [],
            "task_results": RESET,
            "delegation_instructions": RESET,
            "reference_context_ids": ref_ids if ref_ids else None,
            "reference_note": response.reference_note if (response and ref_ids) else None,
            "pending_action": None,
            "hitl_feedback": None,
            "is_final_step": False
        }

    if response.decision == "vision":
        return {
            "query": query,
            "next_nodes": ["vision"],
        }

    # for direct_answer / ambiguous
    return {
        "messages": [AIMessage(content=response.response)],
        "next_nodes": ["FINISH"],
    }


# NODE 1: THE SUPERVISOR ORCHESTRATOR
def supervisor_node(state: AgentState):
    query = state.get("query", "")
    if not query and state.get("messages"):
        query = str(state["messages"][-1].content)

    task_results = deduplicate_task_results(state.get("task_results", []))
    current_plan = state.get("plan", [])
    completed_steps = state.get("completed_steps", [])
    task_results_str = "\n".join(format_task_result(r) for r in task_results) if task_results else "None"

    context = build_supervisor_context(state)

    try:
        hitl_feedback = state.get("hitl_feedback")

        chat_template = ChatPromptTemplate(
            [
                ("system", supervisor_system_prompt),
                ("human",
                 "Original Question: {query}\n\n"
                 "Internal KB: {kb_context}\n\nUser uploaded document summary: {doc_context}\n\n"
                 "Referenced context (from assistant): {reference_note}\n\n"
                 "Current Plan:\n{plan}\n\nCompleted Steps:\n{completed_steps}\n\n"
                 "Task Results so far:\n{task_results}\n\n"
                 "User feedback on the most recently rejected action (if any -- revise your next "
                 "delegation instruction to address this before retrying):\n{hitl_feedback}")
            ],
            input_variables=["query", "kb_context", "doc_context", "reference_note",
                            "plan", "completed_steps", "task_results", "hitl_feedback"]
        )

        chain = chat_template | supervisor_llm
        response: SupervisorDecision = chain.invoke({
            "query": query,
            "plan": "\n".join(f"- {step}" for step in current_plan) if current_plan else "None",
            "completed_steps": "\n".join(f"- {step}" for step in completed_steps) if completed_steps else "None",
            "task_results": task_results_str,
            "hitl_feedback": hitl_feedback or "None",
            **context,
        })                           # type: ignore

        # SAFETY NET: never let the supervisor emit a parallel fan-out when a HITL confirmation is required.
        next_nodes = response.next_nodes
        if response.requires_hitl and len(next_nodes) > 1:
            logging.error(
                f"Supervisor requested parallel next_nodes {next_nodes} with requires_hitl=True; "
                f"collapsing to the first node only."
            )
            next_nodes = next_nodes[:1]

        return {
            "plan": response.plan,
            "completed_steps": response.completed_steps,
            "next_nodes": next_nodes,
            "delegation_instructions": {"__replace__": True, "entries": response.delegation_instructions},
            "doc_context_mode": response.doc_context_mode,
            "requires_hitl": response.requires_hitl,
            "is_final_step": response.final_step,
            "hitl_feedback": None,
            "pending_action": None,
            "draft_answer": None,
        }
    except Exception as e:
        logging.error(f"Error in Supervisor node: {e}", exc_info=True)
        raise AppException(e, sys)


# NODE 2a: WORKSPACE AGENT -- PROPOSE
def workspace_agent(state: AgentState):
    """
    This agent does the non-deterministic LLM call ONCE and commits its output via a normal `return` 
    (checkpointed by the graph -- never replayed). `workspace_confirm` below does the interrupt and 
    only ever reads that already-committed value.
    """
    instruction = state.get("delegation_instructions", {}).get("workspace_agent", "No instruction provided")
    referenced_context = get_referenced_context(state, state.get("reference_context_ids"))

    prior_results = deduplicate_task_results(state.get("task_results", []))
    task_context = "\n".join(format_task_result(r) for r in prior_results) if prior_results else ""

    audience = "user" if state.get("is_final_step") else "supervisor"
    full_instruction = f"AUDIENCE: {audience}\n\nCONTEXT:\n{referenced_context}\n\n{task_context}\n\nINSTRUCTION: {instruction}"

    try:
        agent_selection_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Analyze the request and choose which workspace agent (gmail or calendar, or 'none' if neither "
             "applies) needs to be called to serve this request. Also specify the concrete action it should take."),
            ("human", "{instruction}"),
        ])

        selection: WorkspaceAgentDecission = (agent_selection_prompt | workspace_llm).invoke(
            {"instruction": full_instruction}
        )                                           # type: ignore

        if selection.agent == "none" or selection.agent not in n8n_agents:
            return {
                "task_results": [task_result(
                    agent="workspace_agent",
                    task=instruction,
                    status="error",
                    output="No matching workspace tool (gmail/calendar) could handle this instruction.",
                )],
                "next_nodes": ["supervisor"],
                "hitl_feedback": None,
                "pending_action": None,
            }

        return {
            "pending_action": {
                "agent": selection.agent,
                "action": selection.action,
                "instruction": instruction,
                "full_instruction": full_instruction,
            },
            "next_nodes": ["workspace_confirm"],
        }

    except Exception as e:
        logging.error(f"Error in workspace_agent (propose) node: {e}", exc_info=True)
        return {
            "task_results": [task_result(agent="workspace_agent", task=instruction, status="error", output=str(e))],
            "next_nodes": ["supervisor"],
        }


# NODE 2b: WORKSPACE AGENT -- CONFIRM & EXECUTE
async def workspace_confirm(state: AgentState):
    is_final_step = bool(state.get("is_final_step"))
    requires_hitl = bool(state.get("requires_hitl"))
    pending = state.get("pending_action") or {}

    agent_name = pending.get("agent")
    action = pending.get("action")
    instruction = pending.get("instruction", "No instruction provided")
    full_instruction = pending.get("full_instruction", instruction)

    if agent_name not in n8n_agents:
        # Defensive guard only -- workspace_agent already validated this
        # before committing `pending_action`, so this shouldn't normally fire.
        return {
            "task_results": [task_result(
                agent="workspace_agent", task=instruction, status="error",
                output="Lost track of the proposed workspace action before confirmation."
            )],
            "next_nodes": ["supervisor"],
            "pending_action": None,
        }

    agent = n8n_agents[agent_name]

    try:
        if requires_hitl:
            decision = interrupt({
                "type": "approval",
                "message": f"{agent_name} is about to: {action}. Approve?",
                "pending_action": {"agent": agent_name, "action": action, "instruction": instruction},
            })
            approved, revision_feedback = _parse_resume_decision(decision)

            if not approved:
                return {
                    "messages": [AIMessage(
                        content="Understood -- I won't proceed with that action. Let me know how you'd like it revised."
                    )],
                    "next_nodes": ["supervisor"],
                    "hitl_feedback": revision_feedback or "User rejected the proposed workspace action.",
                    "pending_action": None,
                }

        result = await agent.ainvoke({"instruction": full_instruction})

        update = {
            "task_results": [task_result(
                agent="workspace_agent", task=instruction, status="success", output=result
            )],
            "hitl_feedback": None,
            "pending_action": None,
        }

        if is_final_step:
            update["messages"] = [AIMessage(content=result)]
            update["next_nodes"] = ["FINISH"]
        else:
            update["next_nodes"] = ["supervisor"]

        return update
    
    except GraphInterrupt:
        raise   # let LangGraph's runtime handle the pause

    except Exception as e:
        logging.error(f"Error in workspace_confirm node: {e}", exc_info=True)
        return {
            "task_results": [task_result(agent="workspace_agent", task=instruction, status="error", output=str(e))],
            "next_nodes": ["supervisor"],
            "pending_action": None,
        }


# NODE 3: RESEARCH NODE
async def research_node(state: AgentState):
    instruction = state.get("delegation_instructions", {}).get("research_agent", state.get("query", ""))
    is_final_step = bool(state.get("is_final_step"))
    system_prompt = research_agent_system_prompt
    kb_meta = state.get("internal_kb_meta", {})
    
    if kb_meta.get("available"):
        system_prompt = (
            f"{research_agent_system_prompt}\n\nNOTE: An internal knowledge base is available on topic "
            f"'{kb_meta.get('topic', 'unknown')}'. If the instruction relates to this topic, use the "
            f"`internal_kb_search` tool."
        )

    agent = create_agent(
        model=research_agent_llm,
        tools=mcp_tools,
        system_prompt=system_prompt,
    )
    config = RunnableConfig({"recursion_limit": 14})
    try:
        response = await agent.ainvoke({"messages": [{"role": "user", "content": instruction}]}, config=config)
        result = response["messages"][-1].content

        if is_final_step:
            return {
                "messages": [AIMessage(content=result)],
                "next_nodes": ["FINISH"],
            }
        else:
            return {
                "task_results": [task_result(
                    agent="research_agent", task=instruction, status="success", output=result
                )],
                "next_nodes": ["supervisor"],
            }

    except Exception as e:
        logging.error(f"Error in research node: {e}", exc_info=True)
        return {
            "task_results": [task_result(agent="research_agent", task=instruction, status="error", output=str(e))],
            "next_nodes": ["supervisor"],
        }


# NODE 4: ANSWER NODE
def answer_node(state: AgentState):
    query = state.get("query")
    instruction = state.get("delegation_instructions", {}).get("answer_agent", state.get("query", ""))
    doc_context_mode = state.get("doc_context_mode")
    referenced_context = get_referenced_context(state, state.get("reference_context_ids"))
    feedback = state.get("hitl_feedback")

    try:
        task_results = deduplicate_task_results(state.get("task_results", []))
        web_results = [format_task_result(r) for r in task_results if r.get("agent") == "research_agent"]
        workspace_results = [format_task_result(r) for r in task_results if r.get("agent") == "workspace_agent"]

        if doc_context_mode == "document":
            uploaded_doc = state.get("uploaded_doc")
            prompt = (
                "Uploaded Document:\n{uploaded_doc}\n\n"
                "WEB RESULTS:\n{web_results}\n\n"
                "Referenced Past Turns:\n{referenced_context}\n\n"
                "Instruction from your Supervisor:\n{instruction}\n\n"
                "ORIGINAL USER QUESTION:\n{query}\n\n"
                "PRIOR FEEDBACK (revise accordingly if present):\n{feedback}"
            )
            variables = {
                "uploaded_doc": uploaded_doc or "None",
                "web_results": "\n".join(web_results) if web_results else "None",
                "referenced_context": referenced_context or "None",
                "instruction": instruction,
                "query": query,
                "feedback": feedback or "None",
            }
        else:
            prompt = (
                "WEB SEARCH RESULTS:\n{web_results}\n\n"
                "WORKSPACE RESULTS:\n{workspace_results}\n\n"
                "Referenced Past Turns:\n{referenced_context}\n\n"
                "Instruction from your Supervisor:\n{instruction}\n\n"
                "ORIGINAL USER QUESTION:\n{query}\n\n"
                "PRIOR FEEDBACK (revise accordingly if present):\n{feedback}"
            )
            variables = {
                "web_results": "\n".join(web_results) if web_results else "None",
                "workspace_results": "\n".join(workspace_results) if workspace_results else "None",
                "referenced_context": referenced_context or "None",
                "instruction": instruction,
                "query": query,
                "feedback": feedback or "None",
            }

        chat_template = ChatPromptTemplate(
            [
                ("system", answer_agent_system_prompt),
                ("human", prompt),
            ],
            input_variables=[],
        )

        chain = chat_template | answer_llm
        response: AnswerAgentResponse = chain.invoke(variables)                               #type: ignore
        final_answer = response.final_answer or "I apologize, but I was unable to generate a proper answer."

        return {
            "draft_answer": final_answer,
            "task_results": [task_result(
                agent="answer_agent", task=instruction, status="success", output=final_answer
            )],
            "next_nodes": ["answer_confirm"],
        }

    except Exception as e:
        logging.error(f"Error in answer node: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I apologize, but I encountered an error generating the final answer. Please try again later.")],
            "next_nodes": ["FINISH"],
        }


def answer_confirm(state: AgentState):
    is_final_step = bool(state.get("is_final_step"))
    requires_hitl = bool(state.get("requires_hitl"))
    final_answer = state.get("draft_answer", "")

    if requires_hitl:
        decision = interrupt({
            "type": "approval",
            "message": "Here is the drafted response -- do you approve it? (y/n)",
            "draft": final_answer,
        })
        approved, revision_feedback = _parse_resume_decision(decision)
        if not approved:
            return {
                "messages": [AIMessage(content="Got it -- I won't finalize this version. Let me know what to change.")],
                "next_nodes": ["supervisor"],
                "hitl_feedback": revision_feedback or "User rejected the drafted answer.",
                "draft_answer": None,
            }

    update = {"hitl_feedback": None, "draft_answer": None}                                        
    if is_final_step:
        update["messages"] = [AIMessage(content=final_answer)]                                #type: ignore
        update["next_nodes"] = ["FINISH"]                                                     #type: ignore
    else:
        update["next_nodes"] = ["supervisor"]                                                 #type: ignore
    return update


# NODE 5: VISION NODE
def vision_node(state: AgentState):
    try:
        image_data = state.get("image_data")
        if image_data:
            response = vision_llm.generate_content([vision_agent_system_prompt, state["query"], image_data[0]])
            return {"messages": [AIMessage(content=response.text)]}
        return {"messages": [AIMessage(content="I couldn't find an image to analyze. Please try uploading it again.")]}

    except Exception as e:
        logging.error(f"Error in vision node: {e}", exc_info=True)
        return {"messages": [AIMessage(content="I apologize, but I encountered an error processing the image. Please try again later.")]}


# CONDITIONAL ROUTING FUNCTIONS
def route_from_assistant(state: AgentState) -> str:
    nodes = state.get("next_nodes", ["supervisor"])
    if "FINISH" in nodes:
        return END
    if "vision" in nodes:
        return "vision"
    return "supervisor"

def route_by_next_nodes(state: AgentState) -> list[str]:
    """
    Generic router: reads `next_nodes` from state and either fans out to those node names or terminates.
    """
    nodes = state.get("next_nodes", ["FINISH"])
    if "FINISH" in nodes:
        return [END]
    return nodes


# BUILD GRAPH WORKFLOW
graph = StateGraph(AgentState)

graph.add_node("assistant", assistant_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("research_agent", research_node)
graph.add_node("workspace_agent", workspace_agent)
graph.add_node("workspace_confirm", workspace_confirm)
graph.add_node("answer_agent", answer_node)
graph.add_node("answer_confirm", answer_confirm)
graph.add_node("vision", vision_node)

graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", route_from_assistant)
graph.add_conditional_edges("supervisor", route_by_next_nodes)
graph.add_conditional_edges("research_agent", route_by_next_nodes)
graph.add_conditional_edges("workspace_agent", route_by_next_nodes)
graph.add_conditional_edges("workspace_confirm", route_by_next_nodes)
graph.add_conditional_edges("answer_agent", route_by_next_nodes)
graph.add_conditional_edges("answer_confirm", route_by_next_nodes)
graph.add_edge("vision", END)

ai_agent = None             # compiled by build_agent(), called from AgentRuntime._startup()

async def build_agent():
    global ai_agent
    from src.db_connections.postgres import init_checkpointer
    checkpointer = await init_checkpointer()
    ai_agent = graph.compile(checkpointer=checkpointer)
    
    return ai_agent