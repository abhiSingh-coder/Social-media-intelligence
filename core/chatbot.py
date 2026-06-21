import streamlit as st
from google import genai


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])


def ask_chatbot(question: str, context: str = "") -> str:
    """
    Sends a question (optionally with data context) to Gemini and returns the answer.
    """
    if not question or question.strip() == "":
        return ""

    client = get_client()

    system_instruction = (
        "You are a helpful assistant for a Social Media Intelligence dashboard. "
        "Answer the user's question clearly and concisely based on the context provided, "
        "if any. If no context is given, answer generally about social media analytics."
    )

    prompt = f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {question}" if context else f"{system_instruction}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text