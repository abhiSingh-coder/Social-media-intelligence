import re
import joblib
import streamlit as st

MODEL_PATH = "trained_models/fake_news_classifier.pkl"
VECTORIZER_PATH = "trained_models/fake_news_vectorizer.pkl"


@st.cache_resource
def load_fake_news_artifacts():
    """
    Load the trained XGBoost model and TF-IDF vectorizer once and cache them.
    """
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def clean_text(text: str) -> str:
    """
    Same cleaning logic used during training — must match exactly,
    otherwise the vectorizer will see different patterns than it learned.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_fake_news(text: str) -> dict:
    """
    Predict whether a given text is fake or real news.
    Returns: {"label": "fake"/"real", "confidence": float}
    """
    if not text or text.strip() == "":
        return {"label": "unknown", "confidence": 0.0}

    model, vectorizer = load_fake_news_artifacts()

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    label = "real" if prediction == 1 else "fake"
    confidence = round(max(probabilities) * 100, 2)

    return {"label": label, "confidence": confidence}