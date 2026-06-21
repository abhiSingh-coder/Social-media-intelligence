import streamlit as st
import pandas as pd
from transformers import pipeline


@st.cache_resource
def load_sentiment_pipeline():
    """
    Load the pretrained RoBERTa sentiment model once and cache it.
    Model returns labels: negative, neutral, positive
    """
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for a single piece of text.
    Returns: {"label": "positive"/"negative"/"neutral", "score": float}
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"label": "neutral", "score": 0.0}

    classifier = load_sentiment_pipeline()
    result = classifier(text[:512])[0]  # truncate to avoid token limit issues
    return {"label": result["label"].lower(), "score": round(result["score"], 4)}


def predict_sentiment_batch(texts: list) -> pd.DataFrame:
    """
    Predict sentiment for a list of texts (use on a SMALL sample, not full dataset).
    Returns a dataframe with columns: text, label, score
    """
    classifier = load_sentiment_pipeline()
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