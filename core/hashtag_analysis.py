import re
from itertools import combinations
import pandas as pd


def extract_hashtags(text: str) -> list:
    """
    Extract hashtags from raw (uncleaned) text.
    Note: use the ORIGINAL text column here, not clean_text,
    since clean_text already strips the '#' symbol.
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"#(\w+)", text.lower())


def get_top_hashtags(df: pd.DataFrame, text_column: str = "text", top_n: int = 15) -> pd.DataFrame:
    """
    Returns a dataframe of top N hashtags by frequency.
    """
    all_hashtags = []
    for text in df[text_column].dropna():
        all_hashtags.extend(extract_hashtags(text))

    if not all_hashtags:
        return pd.DataFrame(columns=["hashtag", "count"])

    counts = pd.Series(all_hashtags).value_counts().head(top_n)
    return counts.reset_index().rename(columns={"index": "hashtag", 0: "count"})


def get_hashtag_cooccurrence(df: pd.DataFrame, text_column: str = "text", top_n: int = 20) -> pd.DataFrame:
    """
    For every post with 2+ hashtags, count how often each pair appears together.
    Returns top N pairs as a dataframe with columns: hashtag_1, hashtag_2, count.
    """
    pair_counts = {}

    for text in df[text_column].dropna():
        tags = sorted(set(extract_hashtags(text)))
        if len(tags) < 2:
            continue
        for pair in combinations(tags, 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if not pair_counts:
        return pd.DataFrame(columns=["hashtag_1", "hashtag_2", "count"])

    pairs_df = pd.DataFrame(
        [(a, b, c) for (a, b), c in pair_counts.items()],
        columns=["hashtag_1", "hashtag_2", "count"]
    )
    return pairs_df.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)