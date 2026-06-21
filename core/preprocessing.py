import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import streamlit as st

# Download stopwords once (no-op if already downloaded)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

RAW_DATA_PATH = "data/processed/sentiment140_sample.csv"
COLUMN_NAMES = ["target", "id", "date", "flag", "user", "text"]


@st.cache_data
def load_dataset(path: str = RAW_DATA_PATH, sample_size: int = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="latin-1"
    )

    # Remove timezone abbreviation (e.g. "PDT") before parsing
    df["date"] = df["date"].str.replace(r"\s[A-Z]{3}\s", " ", regex=True)

    # Now parse with explicit format: "Mon Apr 06 22:19:45 2009"
    df["date"] = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S %Y", errors="coerce")

    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean a single piece of text:
    - lowercase
    - remove URLs, mentions, hashtags symbol (keep word), punctuation
    - remove stopwords (optional)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"@\w+", "", text)                   # remove mentions
    text = re.sub(r"#", "", text)                       # keep hashtag word, remove '#'
    text = re.sub(r"[^a-z\s]", "", text)                # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()            # remove extra whitespace

    if remove_stopwords:
        words = text.split()
        words = [w for w in words if w not in STOPWORDS]
        text = " ".join(words)

    return text


def clean_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Apply clean_text to an entire dataframe column.
    Adds a new column 'clean_text'.
    """
    df = df.copy()
    df["clean_text"] = df[text_column].apply(clean_text)
    return df