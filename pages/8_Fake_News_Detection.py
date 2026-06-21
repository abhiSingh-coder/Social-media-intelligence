import streamlit as st
from core.auth import require_login
from core.fake_news_model import predict_fake_news

require_login()

st.title("📰 Fake News Detection")

st.write("Paste a news headline or article snippet below to check if it's likely real or fake.")

user_text = st.text_area("News headline or article text:", height=150)

if st.button("Check Authenticity"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        result = predict_fake_news(user_text)

        if result["label"] == "real":
            st.success(f"✅ Likely **Real News** (confidence: {result['confidence']}%)")
        elif result["label"] == "fake":
            st.error(f"🚨 Likely **Fake News** (confidence: {result['confidence']}%)")
        else:
            st.info("Unable to determine.")

st.caption("Model trained on the Kaggle 'Fake and Real News Dataset' using TF-IDF + XGBoost (99.8% test accuracy).")