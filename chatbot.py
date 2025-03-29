from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "2025-vitap-chatbot"

# Initialize the LLM
llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model_name='gemma2-9b-it')

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create a StateGraph instance
graph_builder = StateGraph(State)

# Define the chatbot function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Add chatbot to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)

# Compile the graph
app = graph_builder.compile()