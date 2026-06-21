import pandas as pd


def get_daily_volume(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Returns a dataframe with post count per day.
    """
    daily = (
        df.groupby(df[date_column].dt.date)
        .size()
        .reset_index(name="post_count")
    )
    daily.columns = ["date", "post_count"]
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def detect_spikes(daily: pd.DataFrame, window: int = 3, threshold: float = 1.5) -> pd.DataFrame:
    """
    Flags days where post_count is significantly above the rolling average.
    threshold: multiplier — e.g. 1.5 means 'volume is 1.5x the rolling average'.
    """
    daily = daily.copy()
    daily["rolling_avg"] = daily["post_count"].rolling(window=window, min_periods=1).mean()
    daily["is_spike"] = daily["post_count"] > (daily["rolling_avg"] * threshold)
    return daily


def get_top_keywords_by_frequency(df: pd.DataFrame, text_column: str = "clean_text", top_n: int = 15) -> pd.DataFrame:
    """
    Simple frequency-based 'what's trending' word list (not TF-IDF — that's a separate module).
    """
    all_words = " ".join(df[text_column].dropna()).split()
    word_counts = pd.Series(all_words).value_counts().head(top_n)
    return word_counts.reset_index().rename(columns={"index": "word", 0: "count"})