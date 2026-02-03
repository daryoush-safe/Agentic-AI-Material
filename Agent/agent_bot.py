from typing import List, TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.environ.get("API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("You: ")
while user_input.lower() not in {"exit", "quit"}:
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("You: ")