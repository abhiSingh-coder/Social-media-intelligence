import re
import joblib
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = STOPWORDS.union({
    "amp", "dont", "cant", "wont", "im", "ive", "youre", "thats",
    "get", "got", "like", "would", "really", "one", "know", "want",
    "go", "going", "still", "much", "lol"
})

MODEL_PATH = "trained_models/lda_model_sklearn.pkl"
VECTORIZER_PATH = "trained_models/lda_vectorizer_sklearn.pkl"
LABELS_PATH = "trained_models/lda_topic_labels.pkl"


@st.cache_resource
def load_topic_model():
    lda_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    topic_labels = joblib.load(LABELS_PATH)
    return lda_model, vectorizer, topic_labels


def clean_for_topics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"&amp;?", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 2]
    return " ".join(words)


def get_topic_distribution(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    lda_model, vectorizer, topic_labels = load_topic_model()

    cleaned_texts = df[text_column].dropna().apply(clean_for_topics)
    cleaned_texts = cleaned_texts[cleaned_texts.str.strip() != ""]

    if cleaned_texts.empty:
        return pd.DataFrame(columns=["text", "topic_id", "topic_label"])

    doc_term_matrix = vectorizer.transform(cleaned_texts)
    topic_distributions = lda_model.transform(doc_term_matrix)

    results = []
    for original_text, dist in zip(df.loc[cleaned_texts.index, text_column], topic_distributions):
        dominant_topic = dist.argmax()
        results.append({
            "text": original_text,
            "topic_id": dominant_topic,
            "topic_label": topic_labels.get(dominant_topic, f"Topic {dominant_topic}")
        })

    return pd.DataFrame(results)


def get_topic_keywords(num_words: int = 8) -> dict:
    lda_model, vectorizer, topic_labels = load_topic_model()
    feature_names = vectorizer.get_feature_names_out()

    output = {}
    for idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-num_words:][::-1]]
        label = topic_labels.get(idx, f"Topic {idx}")
        output[label] = ", ".join(top_words)

    return output