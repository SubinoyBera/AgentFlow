# Multi-Agentic LLM Orchestration
This project implements a scalable agentic AI system that dynamically routes user queries across multiple specialized agents using a graph-based execution model. Built with LangGraph and LangChain, the system demonstrates how modern AI applications can move beyond single-prompt chatbots towards modular, tool-aware, and observable AI workflows.

At the core of the system is a Router Agent that analyzes each user query and, at runtime, decides whether to handle it via real-time Web Search, Retrieval-Augmented Generation (RAG), or process an image to respond the user query. A dedicated agent handles each responsibility, uses external tools, and the final response is synthesized by an Answer Agent, ensuring clarity and reliability.

The entire workflow is orchestrated using LangGraph’s conditional state transitions, making the system easy to extend, debug, and evaluate. A Streamlit-based UI provides an interactive interface, while LangSmith integration enables full observability, tracing, and performance analysis of agent decisions and tool usage.

Modern LLM applications often fail when a single agent is forced to handle
retrieval, web search, reasoning, and response generation simultaneously.

The project explores a modular, **graph-based agentic architecture** where:
- Queries are dynamically routed
- Specialized agents handle distinct responsibilities
- Tools are invoked only when needed
- The entire workflow is observable and debuggable

The goal was to design an *extensible, production-oriented agent system* rather than a single prompt-based chatbot!! 🎓


## 🧩 Agent Workflow
![Agentic Workflow](assets/workflow.png)


## ⚙️ Workflow Overview

1. **Router Agent**
   - Understands and routes incoming queries to other agents.
   - Decides whether RAG, web search, or research is required, else answer.

2. **RAG Agent**
   - Handles knowledge-grounded queries.
   - Uses vector search retrieval from Pinecone vectorstore for contextual answers.

3. **Web Agent**
   - Performs real-time web searches using different search tools.
   - Useful for recent or dynamic information.

4. **Vision Agent**
   - Process and analyze the image uploaded.
   - Answers image-based queries given by the user.

5. **Answer Agent**
   - Synthesizes final output from the available context
   - Produces a final, coherent response


## ⭐ Key Features
- **Stateful Agent workflow**: Implemented the entire workflow as a stateful graph with conditional transitions, enabling clear control flow, extensibility, and debuggability.
- **Multi-Agent Coordination**: Coordinates multiple specialized agents, allowing agents to collaborate, pass intermediate results, and contribute to a unified final response.
- **Tool-Augmented Reasoning with ReAct framework**: Agents can invoke external tools such as web search to enhance factual accuracy and reasoning depth.
- **Agent-Orchestrated Adaptive RAG (Retrieval-Augmented Generation)**: Uses a Router Agent to dynamically decide whether a query requires looking into external database provided, web search, or multi-step research—avoiding unnecessary RAG calls.
- **Separation of Concerns in LLM Systems**: Separates responsibilities such as routing, retrieval, tool usage, and response synthesis across dedicated agents, improving maintainability, scalability, and modular design.
- **Observability with LangSmith**: Full visibility into agent decisions, tool calls, and execution paths for debugging and evaluation.


## 🏗️ Tech Stack
1. **LangGraph** – Defines the agentic workflow as a stateful graph with conditional routing and execution paths.
2. **LangChain** – Provides abstractions for agents, tools, prompts, and LLM interactions.
3. **LangSmith** - End-to-end tracing, debugging, and evaluation of agent decisions, tool calls, and execution paths.
4. **LLMs** – GPT-OSS-120b, Llama, Gemini-2.5-flash, GoogleGenerativeAIEmbeddings
5. **Databases** - Pinecone, SqlLite.
6. **Custom Tools** – Modular tool interfaces for different tasks.
7. **Streamlit** - Interactive web-based UI for real-time user interaction with the agentic system.

<b> 📍 All open source tools and free tier services have been used to develop this application!! 🙏✨</b>


## 🎯 Application Preview

![App](assets/app_img.png)


![LangSmith Tracing](assets/langsmith.png)

## 🛠️ Installation

STEP: 01 - Clone this repository

```cmd
git clone https://github.com/SubinoyBera/AgentFlow
cd ToxicTweet-Tagger
```

STEP: 02 - Create and activate conda environment

```bash
conda create -p venv python=3.11.5 -y
conda activate venv
```

STEP: 03 - Install project requirements. <br>

```bash
pip install -r requirements.txt
```

STEP: 4 - Setup the `.env` file and add your API keys in the environment variables as specified in `.env.example`

STEP: 5 - Run the Application

```bash
streamlit run app.py
```

Open your browser and open the local URL: `http://localhost:8501/`.<br>


## ⚠️ Limitations & Known Constraints

This project is implemented using publicly available, free-tier APIs. As a result, this has certain limitations which are expected and acknowledged:

- **API Rate Limits**  
  Free-tier access to services such as Tavily, Google Serper, etc., and LLM Providers enforces strict request limits, which can affect throughput under frequent usage.

- **Dependency on External Services**  
  System reliability depends on the availability and responsiveness of third-party APIs.

- **Increased Latency**  
  Multi-agent execution combined with external API calls can introduce noticeable latency, particularly when web search or research agents are invoked.

- **Cold Start Delays**  
  Free-tier services may experience cold starts or slower response times, especially after periods of inactivity.


## 🔮 Future Work & Extensions

The agentic architecture used in this project is designed to be extensible and scalable. Several enhancements can be explored in future iterations:

- Addition of more Agents to perform a wide variety of tasks: New agents can be added to the workflow to handle tasks such as email management, calendar scheduling, document summarization, or task automation without disrupting existing agents.

- Memory & Personalization: Add short-term and long-term memory modules to enable personalized and context-aware interactions across sessions.

- Production-Grade Deployment: Replace the Streamlit interface with a scalable backend and API-driven architecture suitable for multi-user environments.

<hr>

🎗️🙏 **THANK YOU !!** :) <br>
<b>-_with love & regards : Subinoy Bera (developer)_</b><br>
🧡🤍💚


