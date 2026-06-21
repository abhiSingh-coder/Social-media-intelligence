import streamlit as st

# Dummy credentials for now — replace with a real check later if needed
VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"


def check_login(username: str, password: str) -> bool:
    """Validate credentials. Returns True if valid."""
    return username == VALID_USERNAME and password == VALID_PASSWORD


def login_user():
    """Mark the current session as logged in."""
    st.session_state["logged_in"] = True


def logout_user():
    """Clear login state."""
    st.session_state["logged_in"] = False


def is_logged_in() -> bool:
    """Check if current session is authenticated."""
    return st.session_state.get("logged_in", False)


def require_login():
    """
    Call this at the top of any protected page.
    Stops page rendering if user isn't logged in.
    """
    if not is_logged_in():
        st.warning("Please log in first.")
        st.stop()