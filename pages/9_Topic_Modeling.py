import streamlit as st
from core.auth import require_login
from core.preprocessing import load_dataset
from core.topic_modeling import get_topic_distribution, get_topic_keywords

require_login()

st.title("🧩 Topic Modeling")
st.caption("Topics discovered using LDA (Latent Dirichlet Allocation) on the Sentiment140 dataset.")

st.subheader("Discovered Topics")
topic_keywords = get_topic_keywords()

for label, keywords in topic_keywords.items():
    st.markdown(f"**{label}**")
    st.caption(keywords)

st.divider()

st.subheader("Topic Distribution Across Sample Posts")
with st.spinner("Assigning topics to sample posts..."):
    df = load_dataset(sample_size=300)
    topic_df = get_topic_distribution(df)

if topic_df.empty:
    st.info("No topics could be assigned to this sample.")
else:
    topic_counts = topic_df["topic_label"].value_counts()
    st.bar_chart(topic_counts)
    st.dataframe(topic_df.head(20))