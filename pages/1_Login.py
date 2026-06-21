import streamlit as st
from core.auth import check_login, login_user, is_logged_in

st.title("Login")

if is_logged_in():
    st.success("You are already logged in.")
    st.write("Go to the Dashboard from the sidebar.")
else:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            login_user()
            st.success("Login successful! Go to Dashboard from the sidebar.")
        else:
            st.error("Invalid username or password.")