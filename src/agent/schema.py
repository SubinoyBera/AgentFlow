from pydantic import BaseModel, Field
from typing import Literal, Optional, TypedDict

# OUTPUT SCHEMA FOR GENERAL ASSISTANT
class AssistantDecision(BaseModel):
    decision: Literal["direct_answer", "ambiguous", "task", "vision"] = Field(
        description=(
            """
            direct_answer: answerable directly and confidently from general knowledge and available context without any research or tools. e.g. greetings/ small talk, summarization, etc.
            ambiguous: genuine fork in meaning that changes the correct answer and cannot be safely guessed without any relevant context.
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
    reference_note: Optional[str] = Field(
        default=None,
        description=(
            "Required whenever reference_context_ids is set, for 'task' only. A short (1-3 sentence) "
            "note the SUPERVISOR will plan from -- it will never see the referenced content itself, "
            "only this note. Summarize what the referenced turn(s) actually contain, and flag anything "
            "that affects planning, e.g. whether it looks current/complete or whether fresh research is "
            "likely needed before acting on it."
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
        default="none",
        description=(
            """
            'document': answer_agent receives the full document as context, if the query is related to the uploaded document and cannot be fully answered from the referenced context.
            'none': answer_agent uses ALL task_results (research, actions, references) as context for answering the query.
            """
        )
    )
    requires_hitl: bool = Field(
        default=False,
        description=(
            "Set True ONLY when the step requires human review/approval before proceeding, otherwise False."
        )
    )
    final_step: bool = Field(
        default=False,
        description=(
            "Set True ONLY when this is the single last step of your plan -- once its result comes back, nothing else needs to happen."
            "Leave False whenever more steps/ actions are still required to complete the plan, and you expect to receive back the results."
        )
    )


# OUTPUT SCHEMA FOR THE WORKSPACE READ-PLANNING STEP
class WorkspaceAgentDecission(BaseModel):
    agent: Literal["gmail", "calendar", "none"] = Field(
        description="Which n8n agent to be called for serving the request. 'none' if the request is does not require gmail or calendar agent."
    )
    action: Optional[str] = Field(
        default=None,
        description=(
            "Required unless domain is 'none'. The specific action instructed to perform. Example: Schedule meeting with Harry today at 5pm, Send email to kane.de@gmail.com, Delete event on 20th June."
        )
    )


# OUTPUT SCHEMA FOR ANSWER AGENT
class AnswerAgentResponse(BaseModel):
    final_answer: Optional[str] = Field(default=None, description="The final answer generated by the answer agent. Set to None if more information is needed.")


class TaskResult(TypedDict, total=False):
    agent: str
    task: str
    status: Literal["success", "error"]
    output: str
    metadata: dict