# AgentFlow

This project implements a production-style agentic AI system that dynamically routes user queries across multiple specialized agents using a graph-based execution model. Built with LangGraph, LangChain, and LangSmith, the system demonstrates how modern LLM applications can move beyond single-prompt chatbots toward modular, tool-aware, and observable AI workflows.

At the core of the system is a Router Agent that analyzes each user query and decides—at runtime—whether it should be handled via Retrieval-Augmented Generation (RAG), real-time web search, or multi-step research using external tools. Each responsibility is handled by a dedicated agent, and the final response is synthesized by an Answer Agent, ensuring clarity and reliability.

The entire workflow is orchestrated using LangGraph’s conditional state transitions, making the system easy to extend, debug, and evaluate. A Streamlit-based UI provides an interactive interface, while LangSmith integration enables full observability, tracing, and performance analysis of agent decisions and tool usage.

Modern LLM applications often fail when a single agent is forced to handle
retrieval, web search, reasoning, and response generation simultaneously.

The project explores a **modular, graph-based agentic architecture** where:
- Queries are dynamically routed
- Specialized agents handle distinct responsibilities
- Tools are invoked only when needed
- The entire workflow is observable and debuggable

The goal was to design an **extensible, production-oriented agent system** rather than a single prompt-based chatbot!!


## Agent Workflow
![Agentic Workflow](assets/workflow.png)


## Workflow Overview

1. **Router Agent**
   - Understands and routes incoming queries to other agents.
   - Decides whether RAG, web search, or research is required, else answer.

2. **RAG Agent**
   - Handles knowledge-grounded queries.
   - Uses vector search retrieval from Pinecone vectorstore for contextual answers.

3. **Web Agent**
   - Performs real-time web searches using different search tools.
   - Useful for recent or dynamic information.

4. **Research Agent**
   - Executes multi-step reasoning and prepares research report on the given topic.
   - Uses external tools for deeper analysis

5. **Answer Agent**
   - Synthesizes final output from the available context
   - Produces a final, coherent response


## 

