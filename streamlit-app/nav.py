"""Page registry and navigation state."""

import streamlit as st

PAGES = ["Overview", "Live Webcam", "Image Analysis", "Video Analysis", "About"]
DEFAULT_PAGE = PAGES[0]


def current_page():
    page = st.session_state.get("page", DEFAULT_PAGE)
    return page if page in PAGES else DEFAULT_PAGE


def go(page):
    """Callback for nav buttons (runs before the rerun, so the new page renders immediately)."""
    st.session_state["page"] = page
