import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
# import gradio as gr


load_dotenv()

model = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.environ.get("API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0
)

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     openai_api_key=os.environ.get("API_KEY"),
#     openai_api_base="https://api.groq.com/openai/v1",
# )
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pdf_path = "./Agent/Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The file {pdf_path} does not exist.")

pdf_loader = PyPDFLoader(pdf_path)
try:
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages from the PDF.")
except Exception as e:
    print(f"Failed to load PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pages_split = text_splitter.split_documents(pages) # apply splitter to all pages

persistent_directory = "./chroma_db"
collection_name = "stock_market_performance_2024"

if not os.path.exists(persistent_directory):
    os.makedirs(persistent_directory)

try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persistent_directory,
        collection_name=collection_name,
    )
    print(f"Vector store created with collection name: {collection_name}")
except Exception as e:
    print(f"Failed to create vector store: {e}")
    raise

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # amount of chunks to retrieve
)

@tool
def retriever_tool(query: str) -> str:
    """Tool to retrieve relevant document chunks based on a query."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    results = [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    return "\n\n".join(results)

tools = [retriever_tool]

model = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> bool:
    """Checks wether las message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state and return the updated state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    response = model.invoke(messages)
    return {'messages': [response]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Function to take action based on the last message's tool calls."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()

# def predict(message, history):
#     # history in Gradio 4 is a list of tuples: [[user, bot], [user, bot]]
#     input_messages = []
#     for user_msg, ai_msg in history:
#         input_messages.append(HumanMessage(content=user_msg))
#         input_messages.append(AIMessage(content=ai_msg))
    
#     input_messages.append(HumanMessage(content=message))
#     result = rag_agent.invoke({"messages": input_messages})
#     return result['messages'][-1].content

# # Create the interface without 'type' or 'theme'
# demo = gr.ChatInterface(
#     fn=predict,
#     title="📈 2024 Stock Market Analyst",
#     description="Ask questions about the 2024 Stock Market Performance PDF.",
#     examples=["How did the S&P 500 perform in Q1?", "Which sectors had the highest growth?"],
# )

# if __name__ == "__main__":
#     # Launch with theme here
#     demo.launch()
