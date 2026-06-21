import streamlit as st
from core.auth import require_login, logout_user

require_login()

st.title("Dashboard")
st.write("This is a placeholder. Module summaries will appear here later.")

if st.button("Logout"):
    logout_user()
    st.success("Logged out. Go to Login page from the sidebar.")