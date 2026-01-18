from typing_extensions import TypedDict, Literal, Annotated
from langchain_core.messages import AIMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import AgentExecutor, create_react_agent 
from pydantic import BaseModel, Field
from ..tools.tools import retriever, tavily_search, news_search, weather_tool, stock_finance_tool
from utils.common import get_checkpointer
from .prompt import (router_system_prompt, rag_agent_system_prompt, web_agent_prompt,
                    answer_agent_prompt, research_agent_prompt)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route_decision: Literal["rag", "web", "answer", "research", "router", "none"]
    external_kb_meta: dict
    rag_results: str
    web_results: str
    intermediate_query: bool
    response: str

class RouteDecission(BaseModel):
    route: Literal["rag", "web", "answer", "research", "none"]
    reply: str = Field(description="A reply message from LLM")

class RagVerdict(BaseModel):
    is_sufficient: bool = Field(..., description="True if retrieved information is sufficient to answer the user query, otherwise False")

class AnswerAgentResponse(BaseModel):
    intermediate_query: bool
    response: str = Field(description="Filled only if intermediate_question is not 'None', otherwise 'None' ")


# Initialize external knowledge base
external_kb_meta = {
    "available": False,
    "topic": None,
    "summary": None
}

# Initialize LLMs
router_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0).with_structured_output(RouteDecission)
rag_agent_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0).with_structured_output(RagVerdict)
web_agent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
answer_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7).with_structured_output(AnswerAgentResponse)


# Node 1 : ROUTER NODE
def router_node(state: AgentState):
    if state.get("intermediate_query", False):
        query = state["response"]
    else:
        query = str(state["messages"][-1].content)
    
    chat_template = ChatPromptTemplate(
        [
            ("system", router_system_prompt),
            ("human", "external_kb_meta: {external_kb_meta}\n\n Question: {query}")
        ],
        input_variables=["external_kb_meta", "query"]
    )
    
    router = chat_template | router_llm 
    response : RouteDecission = router.invoke({"external_kb_meta": external_kb_meta, "query": query})    # type: ignore

    if response.route=="none" and response.reply is not None:
        return {"route_decision": response.route, "response": response.reply, "messages": AIMessage(content=response.reply)}
  
    return {"route_decision": response.route, "query": query}


# NODE 2 : RAG NODE
def rag_node(state: AgentState):
    docs = retriever.invoke({"query": state["query"], "index_name": state["external_kb_meta"]["topic"]})

    if docs.startswith('rag_error'):
        return {"route_decision": "web", "rag_results": None}

    prompt_template = ChatPromptTemplate(
        [
            ("system", rag_agent_system_prompt),
            ("human", "Question: {query}\n\n retrieved_docs: {retreived_docs}")
        ],
        input_variables=["query", "docs"]
    )

    rag = prompt_template | rag_agent_llm
    verdict: RagVerdict = rag.invoke({"query": state["query"], "retreived_docs": docs})                          # type: ignore

    # Decide next route
    if verdict.is_sufficient:
        return {"route_decision": "answer", "rag_results": docs}
    else:
        return {"route_decision": "web", "rag_results": "Sufficient information is not available"}


# NODE 3: WEB SEARCH NODE
def web_node(state: AgentState):
    # after web agent always redirect to answer agent
    state["route_decision"] = "answer"

    web_agent = create_react_agent(
        llm = web_agent_llm,
        tools = [tavily_search, news_search, weather_tool, stock_finance_tool],
        prompt = PromptTemplate(template=web_agent_prompt, input_variables=["query", "agent_scratchpad"])
    )

    agent_executor = AgentExecutor(
        agent = web_agent,
        tools = [tavily_search, news_search, weather_tool, stock_finance_tool],
        max_iterations=5,
        max_execution_time=25,
        handle_parsing_errors=True,
        early_stopping_method="force"
    )

    response = agent_executor.invoke({"query": state["query"]})
    return {"web_results": response["output"]}


# NODE 5: RESEARCH NODE
def research_node(state: AgentState):
    tools = [tavily_search, news_search]
    research_agent = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5).bind_tools(tools)

    prompt_template = ChatPromptTemplate(
        [
            ("system", "You are a senior expert researcher. Your task is to perform **extensive research** and prepare a detailed **professional** research report on the given topic. Make sure to use all your experiences and research skills. Provide citations, and add a precise conclusion highlighting the key findings, insights and final results."),
            ("human", research_agent_prompt)
        ],
        input_variables = ["topic", "agent_scratchpad"]
    )

    chain = prompt_template | research_agent | StrOutputParser()
    research_report = chain.invoke({"topic": state["query"], "tools": tools})

    #chat_history.append(AIMessage(research_report))
    return {"final_response": research_report}


# NODE 5: RESEARCH TOOLS NODE
research_tools = [tavily_search, news_search]
tool_node = ToolNode(research_tools)


# NODE 6: ANSWER NODE
def answer_node(state: AgentState):
    rag_results = state.get("rag_results", "None")
    web_results = state.get("web_results", "None")

    try:
        chat_history = state["messages"][-15:]
        parser = PydanticOutputParser(pydantic_object=AnswerAgentResponse)
        chat_template = ChatPromptTemplate(
        [
            ("system", answer_agent_prompt),
            ("human", "Context: {rag_results}\n\n Web Results: {web_results}\n\n Chat History: {chat_history}\n\n Question: {query}")
        ],
        input_variables=["rag_results", "web_results", "chat_history", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = chat_template | answer_llm | parser
        response : AnswerAgentResponse=chain.invoke({"query": state["query"], "rag_results": rag_results, 
                                                     "web_results": web_results, "chat_history": chat_history})
        
        if response.intermediate_query:
            next_route = "router"
            return {"route_decision": next_route, "intermediate_query": response.intermediate_query, "response": response.response}
        else:
            next_route = "none"
            return{"route_decision": next_route, "response": response.response, 
                   "intermediate_query": response.intermediate_query, "messages": AIMessage(content=response.response)}
        
    except Exception as e:
        return {"messages": [AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")]}


def from_router(state) -> Literal["rag", "web", "answer", "research", "none"]:        
    if state["route_decision"] == "router":
        raise ValueError("route_decision from router node cannot be 'router' again")
    return state["route_decision"]

def after_rag(state) -> Literal["web_agent", "answer_agent"]:
    if state["route_decision"] == "web":
        return "web_agent"
    else:
        return "answer_agent"

def from_answer(state) -> Literal["follow_up", "none"]:
    if state["route_decision"] == "router":
        return "follow_up"
    else:
        return "none"
    

# BUILD GRAPH WORKFLOW :
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("rag_agent", rag_node)
graph.add_node("web_agent", web_node)
graph.add_node("answer_agent", answer_node)
graph.add_node("research_agent", research_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("router")
graph.add_conditional_edges(
    "router",
    from_router,
    {
        "rag": "rag_agent",
        "web": "web_agent",
        "answer": "answer_agent",
        "research": "research_agent",
        "none": END
    }
)
graph.add_conditional_edges(
    "rag_agent",
    after_rag,
    {
        "web_agent": "web_agent",
        "answer_agent": "answer_agent"
    }
)
graph.add_edge("web_agent", "answer_agent")
graph.add_conditional_edges("research_agent", tools_condition)
graph.add_edge("tools", "research_agent")

graph.add_edge("research_agent", END)
graph.add_conditional_edges(
    "answer_agent",
    from_answer,
    {
        "follow_up": "router",
        "none": END
    })

checkpointer = get_checkpointer()

ai_agent = graph.compile(checkpointer=checkpointer)