import sys
import os

# Add src to python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../../d:/My Projects/Multi-AI Agent")))

from src.agent.langgraph_agent import ai_agent
from langchain_core.messages import HumanMessage

# Run the agent
inputs = {
    "messages": [HumanMessage(content="Compare the latest stock prices of Amazon and Nvidia")],
    "query": "Compare the latest stock prices of Amazon and Nvidia",
    "external_kb_meta": {"available": False}
}

config = {"configurable": {"thread_id": "test_thread_1"}}

print("Starting agent...")
for event in ai_agent.stream(inputs, config=config):
    for node, state in event.items():
        print(f"\n=== Node: {node} ===")
        if "next_nodes" in state:
            print(f"Next nodes decided by supervisor: {state['next_nodes']}")
        if "delegation_instructions" in state:
            print(f"Delegation instructions: {state['delegation_instructions']}")
        if "task_results" in state:
            print(f"Task results count: {len(state['task_results'])}")
            for res in state['task_results']:
                print(f" - {res[:200]}...")
        if "messages" in state:
            print(f"Last message: {state['messages'][-1].content}")
