import streamlit as st
from core.auth import require_login
from core.chatbot import ask_chatbot

require_login()

st.title("🤖 Chatbot")
st.write("Ask a question about social media trends, sentiment, or this dashboard.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_question = st.text_input("Your question:")

if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_chatbot(user_question)
        st.session_state["chat_history"].append((user_question, answer))

for q, a in reversed(st.session_state["chat_history"]):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.divider()