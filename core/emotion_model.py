import streamlit as st
import pandas as pd
from transformers import pipeline


@st.cache_resource
def load_emotion_pipeline():
    """
    Load the pretrained emotion classification model once and cache it.
    Model returns labels like: joy, anger, sadness, fear, surprise, disgust, neutral
    """
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )


def predict_emotion(text: str) -> dict:
    """
    Predict emotion for a single piece of text.
    Returns: {"label": "joy"/"anger"/etc, "score": float}
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"label": "neutral", "score": 0.0}

    classifier = load_emotion_pipeline()
    result = classifier(text[:512])[0]
    return {"label": result["label"].lower(), "score": round(result["score"], 4)}


def predict_emotion_batch(texts: list) -> pd.DataFrame:
    """
    Predict emotion for a list of texts (use on a SMALL sample, not full dataset).
    Returns a dataframe with columns: text, label, score
    """
    classifier = load_emotion_pipeline()
    results = []

    for text in texts:
        if not isinstance(text, str) or text.strip() == "":
            continue
        prediction = classifier(text[:512])[0]
        results.append({
            "text": text,
            "label": prediction["label"].lower(),
            "score": round(prediction["score"], 4)
        })

    return pd.DataFrame(results)