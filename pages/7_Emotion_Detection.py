import streamlit as st
from core.auth import require_login
from core.preprocessing import load_dataset
from core.emotion_model import predict_emotion, predict_emotion_batch

require_login()

st.title("🎭 Emotion Detection")

# --- Aggregate chart over a small sample ---
st.subheader("Emotion Distribution (Sample)")

with st.spinner("Loading data and running emotion model on a sample..."):
    df = load_dataset(sample_size=200)
    results_df = predict_emotion_batch(df["text"].tolist())

if results_df.empty:
    st.info("No results to display.")
else:
    emotion_counts = results_df["label"].value_counts()
    st.bar_chart(emotion_counts)
    st.dataframe(results_df.head(20))

st.divider()

# --- Live single-text demo ---
st.subheader("Try It Yourself")
user_text = st.text_area("Enter any text to detect its emotion:")

if st.button("Detect Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        result = predict_emotion(user_text)
        st.success(f"**Emotion:** {result['label'].capitalize()} (confidence: {result['score']*100:.1f}%)")