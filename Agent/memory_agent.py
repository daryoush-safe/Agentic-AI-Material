from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.environ.get("API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
)

def process(state: AgentState) -> AgentState:
    """This node is solving the request input by the user."""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    if len(state["messages"]) > 3: # TODO: make this logic more robust
        state["messages"] = state["messages"][1:4]  # Keep only the last 3 messages
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history: List[Union[HumanMessage, AIMessage]] = []

user_input = input("You: ")
while user_input.lower() not in {"exit", "quit"}:
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("You: ")

with open("logging.txt", "w") as file:
    file.write("Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("End of Conversation\n")

print("Conversation logged to logging.txt")