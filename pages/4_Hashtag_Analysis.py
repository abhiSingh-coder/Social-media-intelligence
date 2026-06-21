import streamlit as st
from core.auth import require_login
from core.preprocessing import load_dataset
from core.hashtag_analysis import get_top_hashtags, get_hashtag_cooccurrence

require_login()

st.title("🏷️ Hashtag Analysis")

with st.spinner("Loading data..."):
    df = load_dataset(sample_size=20000)

st.subheader("Top Hashtags")
top_hashtags = get_top_hashtags(df)

if top_hashtags.empty:
    st.info("No hashtags found in this sample. Try increasing sample_size.")
else:
    st.bar_chart(top_hashtags.set_index("hashtag"))

st.subheader("Hashtag Co-occurrence (Pairs Used Together)")
cooccurrence = get_hashtag_cooccurrence(df)

if cooccurrence.empty:
    st.info("No co-occurring hashtag pairs found in this sample.")
else:
    st.dataframe(cooccurrence)