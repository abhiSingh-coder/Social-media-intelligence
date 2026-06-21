import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


@st.cache_resource
def load_summarization_model():
    """
    Load tokenizer and model directly (bypassing pipeline() due to task registry issues
    in some transformers versions).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def summarize_text(text: str, max_length: int = 100, min_length: int = 20) -> str:
    """
    Summarize a longer piece of text into a shorter version.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""

    word_count = len(text.split())
    if word_count < 30:
        return "Text too short to summarize meaningfully."

    tokenizer, model = load_summarization_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary