import streamlit as st
from core.auth import require_login
from core.preprocessing import load_dataset
from core.influencer_ranking import get_top_influencers

require_login()

st.title("🏆 Influencer Ranking")
st.caption(
    "Ranking based on a weighted score combining followers, engagement rate, and posting activity. "
    "Note: follower and engagement figures are simulated for demo purposes since this dataset doesn't "
    "include real social metadata — post counts are the only real metric, derived from actual data."
)

with st.spinner("Computing influencer rankings..."):
    df = load_dataset(sample_size=20000)
    top_influencers = get_top_influencers(df)

if top_influencers.empty:
    st.info("No influencer data available.")
else:
    st.subheader("Top Ranked Users")
    st.dataframe(top_influencers, use_container_width=True)

    st.subheader("Influence Score Comparison")
    st.bar_chart(top_influencers.set_index("user")["influence_score"])