import streamlit as st
from core.auth import require_login
from core.preprocessing import load_dataset
from core.sentiment_model import predict_sentiment, predict_sentiment_batch

require_login()

st.title("😊 Sentiment Analysis")

# --- Aggregate chart over a small sample ---
st.subheader("Sentiment Distribution (Sample)")

with st.spinner("Loading data and running sentiment model on a sample..."):
    df = load_dataset(sample_size=200)  # small sample — transformer inference is slower than pandas ops
    results_df = predict_sentiment_batch(df["text"].tolist())

if results_df.empty:
    st.info("No results to display.")
else:
    sentiment_counts = results_df["label"].value_counts()
    st.bar_chart(sentiment_counts)
    st.dataframe(results_df.head(20))

st.divider()

# --- Live single-text demo ---
st.subheader("Try It Yourself")
user_text = st.text_area("Enter any text to analyze its sentiment:")

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        result = predict_sentiment(user_text)
        st.success(f"**Sentiment:** {result['label'].capitalize()} (confidence: {result['score']*100:.1f}%)")