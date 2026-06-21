import streamlit as st

st.set_page_config(
    page_title="Social Media Intelligence System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Intelligence System")
st.caption("AI-powered analysis of social media data using NLP, Transformers, and Machine Learning")

st.markdown("""
Welcome — this system analyzes social media text data using a combination of 
classical ML models and pretrained transformer models (BERT, RoBERTa) to surface 
trends, sentiment, emotion, misinformation signals, and more.

👉 **Log in from the sidebar to get started.**
""")

st.divider()

st.subheader("What's inside")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📈 Analytics")
    st.markdown("""
    - Trend Analysis
    - Hashtag Analysis
    - Keyword Analytics
    """)

with col2:
    st.markdown("##### 🧠 AI / NLP")
    st.markdown("""
    - Sentiment Analysis (RoBERTa)
    - Emotion Detection (BERT)
    - Fake News Detection (XGBoost)
    - Topic Modeling (LDA)
    """)

with col3:
    st.markdown("##### ✨ Extras")
    st.markdown("""
    - Influencer Ranking
    - Text Summarization
    - AI Chatbot
    - Content Recommendation
    """)

st.divider()

st.caption("Built with Streamlit · scikit-learn · Hugging Face Transformers · Gensim · XGBoost")