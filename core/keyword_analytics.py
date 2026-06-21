import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_top_keywords_tfidf(df: pd.DataFrame, text_column: str = "clean_text", top_n: int = 20) -> pd.DataFrame:
    """
    Extract top N keywords across the dataset using TF-IDF scoring.
    Returns a dataframe with columns: keyword, score.
    """
    documents = df[text_column].dropna()
    documents = documents[documents.str.strip() != ""]

    if documents.empty:
        return pd.DataFrame(columns=["keyword", "score"])

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Average TF-IDF score per word across all documents
    scores = tfidf_matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    keyword_scores = pd.DataFrame({
        "keyword": feature_names,
        "score": scores
    })

    return keyword_scores.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)