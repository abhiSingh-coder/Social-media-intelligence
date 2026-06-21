import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def render_wordcloud(data: dict, width: int = 800, height: int = 400):
    """
    Renders a word cloud given a dictionary of {word: weight}.
    """
    if not data:
        st.info("Not enough data to generate a word cloud.")
        return

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(data)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def dataframe_to_freq_dict(df, word_column: str = "keyword", score_column: str = "score") -> dict:
    """
    Converts a dataframe with word/score columns into a {word: score} dict
    expected by generate_from_frequencies.
    """
    if df.empty:
        return {}
    return dict(zip(df[word_column], df[score_column]))