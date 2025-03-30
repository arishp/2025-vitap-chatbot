from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

st.title('My Chatbot')

user_text = st.text_input("How may I help you?")

messages = [
    (
        "system",
        "You are a helpful assistant that answers my questions",
    ),
    ("human", {user_text}),
]

ai_msg = llm.invoke(messages)
st.write(ai_msg.content)