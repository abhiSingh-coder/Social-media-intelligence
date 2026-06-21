import numpy as np
import pandas as pd


def generate_simulated_metrics(df: pd.DataFrame, user_column: str = "user", seed: int = 42) -> pd.DataFrame:
    """
    Aggregates per-user post count from the dataset (real),
    then simulates follower count and engagement rate (fake, for demo purposes only)
    since this dataset doesn't include real social metadata.
    """
    rng = np.random.default_rng(seed)

    user_stats = df.groupby(user_column).size().reset_index(name="post_count")

    user_stats["followers"] = rng.integers(100, 100000, size=len(user_stats))
    user_stats["avg_engagement_rate"] = rng.uniform(0.5, 12.0, size=len(user_stats)).round(2)  # percentage

    return user_stats


def normalize(series: pd.Series) -> pd.Series:
    """
    Min-max normalize a column to 0-1 range so metrics of different scales
    don't dominate the combined score unfairly.
    """
    if series.max() == series.min():
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def compute_influencer_scores(
    user_stats: pd.DataFrame,
    weight_followers: float = 0.4,
    weight_engagement: float = 0.4,
    weight_activity: float = 0.2
) -> pd.DataFrame:
    """
    Combines normalized followers, engagement, and post activity into a single weighted score.
    """
    df = user_stats.copy()

    df["norm_followers"] = normalize(df["followers"])
    df["norm_engagement"] = normalize(df["avg_engagement_rate"])
    df["norm_activity"] = normalize(df["post_count"])

    df["influence_score"] = (
        weight_followers * df["norm_followers"] +
        weight_engagement * df["norm_engagement"] +
        weight_activity * df["norm_activity"]
    ).round(4)

    return df.sort_values("influence_score", ascending=False).reset_index(drop=True)


def get_top_influencers(df: pd.DataFrame, user_column: str = "user", top_n: int = 15) -> pd.DataFrame:
    """
    Full pipeline: simulate metrics, compute scores, return top N ranked.
    """
    user_stats = generate_simulated_metrics(df, user_column=user_column)
    scored = compute_influencer_scores(user_stats)
    return scored.head(top_n)[
        ["user", "followers", "avg_engagement_rate", "post_count", "influence_score"]
    ]