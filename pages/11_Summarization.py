import streamlit as st
from core.auth import require_login
from core.summarization import summarize_text

require_login()

st.title("📝 Text Summarization")
st.write("Paste a longer post, thread, or article to get a short summary.")

user_text = st.text_area("Text to summarize:", height=200)

if st.button("Summarize"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text(user_text)
        st.success("**Summary:**")
        st.write(summary)