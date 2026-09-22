"""
Theme management component.
Handles dark/light mode toggling and applies appropriate CSS variables.
"""
import streamlit as st
from config import DARK_BG, DARK_CARD, LIGHT_BG, LIGHT_CARD


def init_theme():
    """Initialize theme state in session."""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


def toggle_theme():
    """Toggle between dark and light themes."""
    current = st.session_state.theme
    st.session_state.theme = "dark" if current == "light" else "light"


def get_theme():
    """Get the current theme."""
    return st.session_state.get("theme", "light")


def is_dark():
    """Check if current theme is dark."""
    return get_theme() == "dark"


def apply_theme_css():
    """
    Apply theme CSS variables to the page.
    This modifies the root element's data-theme attribute.
    """
    theme = get_theme()
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {'#121212' if theme == 'dark' else '#FAFAFA'} !important;
            }}
            .stApp header {{
                background-color: {'#1E1E1E' if theme == 'dark' else '#FFFFFF'} !important;
            }}
            .stSidebar {{
                background: {'#0D0D1A' if theme == 'dark' else '#1B1B2F'} !important;
            }}
            .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {{
                color: {'#E0E0E0' if theme == 'dark' else '#212121'} !important;
            }}
            .stSelectbox label, .stSlider label, .stMultiselect label {{
                color: {'#BDBDBD' if theme == 'dark' else '#616161'} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle():
    """Render a theme toggle button in the sidebar."""
    theme = get_theme()
    icon = "🌙" if theme == "light" else "☀️"
    label = "Dark Mode" if theme == "light" else "Light Mode"

    if st.sidebar.button(f"{icon} {label}", key="theme_toggle", use_container_width=True):
        toggle_theme()
        st.rerun()
