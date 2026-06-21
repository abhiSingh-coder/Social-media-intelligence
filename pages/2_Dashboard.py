import streamlit as st
from core.auth import require_login, logout_user
from core.preprocessing import load_dataset
from core.trend_analysis import get_daily_volume, detect_spikes
from core.hashtag_analysis import get_top_hashtags
from core.keyword_analytics import get_top_keywords_tfidf
from core.sentiment_model import predict_sentiment_batch
from core.emotion_model import predict_emotion_batch
from core.influencer_ranking import get_top_influencers
from core.preprocessing import load_dataset, clean_dataframe
from components.wordcloud_chart import render_wordcloud, dataframe_to_freq_dict
require_login()

st.title("📊 Dashboard Overview")
st.caption("A snapshot across all analysis modules. Visit individual pages from the sidebar for full detail.")

with st.spinner("Loading dashboard data..."):
    df_large = load_dataset(sample_size=20000)   # for stats-based modules (fast)
    df_large = clean_dataframe(df_large)     
    df_small = load_dataset(sample_size=150)      # for transformer-based modules (slower)

# Row 1: Trend + Sentiment
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Trending Activity")
    daily = get_daily_volume(df_large)
    daily = detect_spikes(daily)
    st.line_chart(daily.set_index("date")[["post_count", "rolling_avg"]])

with col2:
    st.subheader("😊 Sentiment Distribution")
    sentiment_results = predict_sentiment_batch(df_small["text"].tolist())
    if not sentiment_results.empty:
        st.bar_chart(sentiment_results["label"].value_counts())

# Row 2: Word Cloud-style keywords + Emotion
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔑 Top Keywords")
    top_keywords = get_top_keywords_tfidf(df_large)
    freq_dict = dataframe_to_freq_dict(top_keywords)
    render_wordcloud(freq_dict, width=500, height=250)

with col4:
    st.subheader("🎭 Emotion Breakdown")
    emotion_results = predict_emotion_batch(df_small["text"].tolist())
    if not emotion_results.empty:
        st.bar_chart(emotion_results["label"].value_counts())

# Row 3: Top Influencers + Top Hashtags
col5, col6 = st.columns(2)

with col5:
    st.subheader("🏆 Top Influencers")
    top_influencers = get_top_influencers(df_large, top_n=5)
    if not top_influencers.empty:
        st.dataframe(top_influencers[["user", "influence_score"]], use_container_width=True)

with col6:
    st.subheader("🏷️ Top Hashtags")
    top_hashtags = get_top_hashtags(df_large, top_n=10)
    if top_hashtags.empty:
        st.info("No hashtags found in this sample.")
    else:
        st.bar_chart(top_hashtags.set_index("hashtag"))

st.divider()
if st.button("Logout"):
    logout_user()
    st.success("Logged out. Go to Login page from the sidebar.")