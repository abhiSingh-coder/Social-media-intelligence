import streamlit as st
from core.auth import require_login
from core.preprocessing import load_dataset, clean_dataframe
from core.keyword_analytics import get_top_keywords_tfidf
from components.wordcloud_chart import render_wordcloud, dataframe_to_freq_dict

require_login()

st.title("🔑 Keyword Analytics")

with st.spinner("Loading and cleaning data..."):
    df = load_dataset(sample_size=20000)
    df = clean_dataframe(df)

st.subheader("Top Keywords (TF-IDF Ranked)")
top_keywords = get_top_keywords_tfidf(df)

if top_keywords.empty:
    st.info("No keywords found.")
else:
    st.bar_chart(top_keywords.set_index("keyword"))
    st.dataframe(top_keywords)

    st.subheader("Keyword Word Cloud")
    freq_dict = dataframe_to_freq_dict(top_keywords, word_column="keyword", score_column="score")
    render_wordcloud(freq_dict)