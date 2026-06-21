import re
import joblib
import pandas as pd
import streamlit as st
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = STOPWORDS.union({
    "amp", "dont", "cant", "wont", "im", "ive", "youre", "thats",
    "get", "got", "like", "would", "really", "one", "know", "want",
    "go", "going", "still", "much", "lol"
})

MODEL_PATH = "trained_models/lda_model.gensim"
DICTIONARY_PATH = "trained_models/lda_dictionary.gensim"
LABELS_PATH = "trained_models/lda_topic_labels.pkl"


@st.cache_resource
def load_topic_model():
    """
    Load the trained LDA model, dictionary, and topic labels once and cache them.
    """
    lda_model = LdaModel.load(MODEL_PATH)
    dictionary = corpora.Dictionary.load(DICTIONARY_PATH)
    topic_labels = joblib.load(LABELS_PATH)
    return lda_model, dictionary, topic_labels


def clean_for_topics(text: str) -> list:
    """
    Same cleaning logic used during training — must match exactly
    for the dictionary/model to interpret tokens correctly.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"&amp;?", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 2]
    return words


def get_topic_distribution(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Assigns each document its dominant topic.
    Returns a dataframe with columns: text, topic_id, topic_label
    """
    lda_model, dictionary, topic_labels = load_topic_model()

    results = []
    for text in df[text_column].dropna():
        tokens = clean_for_topics(text)
        if not tokens:
            continue
        bow = dictionary.doc2bow(tokens)
        if not bow:
            continue
        topic_probs = lda_model.get_document_topics(bow)
        if not topic_probs:
            continue
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
        results.append({
            "text": text,
            "topic_id": dominant_topic,
            "topic_label": topic_labels.get(dominant_topic, f"Topic {dominant_topic}")
        })

    return pd.DataFrame(results)


def get_topic_keywords(num_words: int = 8) -> dict:
    """
    Returns the top keywords for each topic, with their human-readable label.
    """
    lda_model, dictionary, topic_labels = load_topic_model()
    output = {}
    for topic_id, topic_str in lda_model.print_topics(num_words=num_words):
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        output[label] = topic_str
    return output