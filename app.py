"""
🍄 Mushroom Classifier - Production-Grade AI Application
=======================================================
Main entry point for the Streamlit application.
Handles page routing, sidebar navigation, and global configuration.

Author: ML Engineering Team
Version: 2.0.0
"""

import streamlit as st
import sys
from pathlib import Path

# ── Ensure project root is in Python path ─────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── Page Configuration (must be the first Streamlit command) ──────
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, INITIAL_SIDEBAR_STATE

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

# ── Global CSS ────────────────────────────────────────────────────
def load_global_css():
    """Load and inject global CSS styles."""
    css_path = ROOT_DIR / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback minimal styles
        st.markdown("""
        <style>
            .card { background: white; border-radius: 12px; padding: 1.5rem; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem; }
            .hero-section { text-align: center; padding: 3rem 1rem; 
                          background: linear-gradient(135deg, #1B5E20, #2E7D32, #388E3C);
                          border-radius: 12px; color: white; margin-bottom: 2rem; }
        </style>
        """, unsafe_allow_html=True)

load_global_css()

# ── Initialize session state ──────────────────────────────────────
from components.sidebar import init_navigation

init_navigation()

# Initialize other session state variables
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

# ── Render Sidebar Navigation ─────────────────────────────────────
from components.sidebar import render_sidebar, get_current_page
from components.theme import apply_theme_css

# Apply theme CSS
apply_theme_css()

# Render sidebar and handle navigation
selected_page = render_sidebar()

# If a page was selected, update the current page
if selected_page:
    st.session_state["current_page"] = selected_page

# Get current page
current_page = get_current_page()

# ── Page Router ────────────────────────────────────────────────────
def render_page(page_name: str):
    """Render the appropriate page based on the route."""
    
    page_routes = {
        "landing": "pages.landing",
        "manual_prediction": "pages.manual_prediction",
        "image_prediction": "pages.image_prediction",
        "analytics": "pages.analytics",
        "history": "pages.history",
        "encyclopedia": "pages.encyclopedia",
        "settings": "pages.settings",
        "about": "pages.about",
    }
    
    if page_name in page_routes:
        module_path = page_routes[page_name]
        try:
            # Import and run the page module
            import importlib
            module = importlib.import_module(module_path)
            if hasattr(module, "app"):
                module.app()
            else:
                st.error(f"Page '{page_name}' does not have an 'app()' function.")
        except ModuleNotFoundError as e:
            st.error(f"Page module not found: {module_path}. Error: {e}")
            # Show landing as fallback
            try:
                import pages.landing as landing
                landing.app()
            except Exception:
                st.info("Welcome to the Mushroom Classifier! Use the sidebar to navigate.")
        except Exception as e:
            st.error(f"Error loading page '{page_name}': {str(e)}")
            import traceback
            st.exception(e)
    else:
        st.warning(f"Unknown page: '{page_name}'. Redirecting to Home.")
        st.session_state["current_page"] = "landing"
        st.rerun()

# ── Render the selected page ──────────────────────────────────────
render_page(current_page)

# ── Footer ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        <p>
            🍄 <strong>Mushroom Classifier</strong> v2.0.0 | 
            Built with ❤️ using Streamlit, Scikit-learn &amp; TensorFlow |
            <a href="#" onclick="alert('Not for actual consumption decisions!')">Disclaimer</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
