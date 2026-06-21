import streamlit as st
from core.auth import require_login
from core.preprocessing import load_dataset, clean_dataframe
from core.trend_analysis import get_daily_volume, detect_spikes, get_top_keywords_by_frequency

require_login()

st.title("📈 Trend Analysis")

with st.spinner("Loading data..."):
    df = load_dataset(sample_size=20000)
    df = clean_dataframe(df)

st.subheader("Post Volume Over Time")
daily = get_daily_volume(df)
daily_with_spikes = detect_spikes(daily)

st.line_chart(daily_with_spikes.set_index("date")[["post_count", "rolling_avg"]])

st.subheader("🔥 Spike Days (Trending Activity)")
spikes = daily_with_spikes[daily_with_spikes["is_spike"]]
if spikes.empty:
    st.info("No significant spikes detected in this sample.")
else:
    st.dataframe(spikes[["date", "post_count", "rolling_avg"]])

st.subheader("Top Trending Words")
top_words = get_top_keywords_by_frequency(df)
st.bar_chart(top_words.set_index("word"))