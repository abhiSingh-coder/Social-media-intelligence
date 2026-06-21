import streamlit as st
from core.auth import require_login
from core.preprocessing import load_dataset, clean_dataframe
from core.recommendation import get_similar_posts

require_login()

st.title("🔁 Recommendation")
st.write("Select a post below to find other posts with similar content.")

with st.spinner("Loading data..."):
    df = load_dataset(sample_size=300)
    df = clean_dataframe(df)

selected_text = st.selectbox("Choose a post:", df["text"].tolist())

if st.button("Find Similar Posts"):
    selected_index = df[df["text"] == selected_text].index[0]
    with st.spinner("Finding similar posts..."):
        similar_posts = get_similar_posts(df, selected_index)

    if similar_posts.empty:
        st.info("No similar posts found.")
    else:
        st.subheader("Similar Posts")
        st.dataframe(similar_posts, use_container_width=True)