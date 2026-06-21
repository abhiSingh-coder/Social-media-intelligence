import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_similar_posts(df: pd.DataFrame, selected_index: int, text_column: str = "clean_text", top_n: int = 5) -> pd.DataFrame:
    """
    Given a dataframe and the index of a selected post, find the top N most similar posts
    based on TF-IDF + cosine similarity.
    """
    documents = df[text_column].fillna("")

    if documents.empty or selected_index not in df.index:
        return pd.DataFrame(columns=["text", "similarity_score"])

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    selected_vector = tfidf_matrix[df.index.get_loc(selected_index)]
    similarity_scores = cosine_similarity(selected_vector, tfidf_matrix).flatten()

    df = df.copy()
    df["similarity_score"] = similarity_scores

    # Exclude the post itself, sort by similarity descending
    results = df[df.index != selected_index].sort_values("similarity_score", ascending=False).head(top_n)

    return results[["text", "similarity_score"]].reset_index(drop=True)