"""
Professional sidebar navigation component.
Provides a modern navigation menu with icons, active state, and branding.
"""
import streamlit as st
from config import APP_NAME, APP_VERSION, PAGES


def render_sidebar():
    """
    Render the main sidebar navigation.
    Returns the selected page path.
    """
    with st.sidebar:
        # ── Branding ───────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🍄</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">Mushroom</div>
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7);">Classifier</div>
                <div style="font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: 0.25rem;">v{APP_VERSION}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Navigation Pages ───────────────────────────────────────
        selected_page = None

        for page in PAGES:
            page_name = page["name"]
            page_icon = page["icon"]
            page_path = page["path"]

            # Determine if this page is currently active
            current_page = st.session_state.get("current_page", "landing")
            is_active = current_page == page_path

            # Active state styling
            active_style = (
                "background: rgba(76, 175, 80, 0.2); "
                "border-left: 3px solid #4CAF50; "
                "color: #FFFFFF;"
                if is_active
                else "color: rgba(255,255,255,0.7);"
            )

            # Render the nav button
            if st.sidebar.button(
                f"{page_icon}  {page_name}",
                key=f"nav_{page_path}",
                use_container_width=True,
                help=f"Go to {page_name}",
            ):
                selected_page = page_path

            # Apply hover style via markdown
            st.markdown(
                f"""
                <style>
                    div[data-testid="stSidebarNav"] div[data-testid="stSidebarNavItems"] 
                    div[data-testid="stSidebarNavItem"]:has(button[key="nav_{page_path}"]) {{
                        {active_style}
                    }}
                    button[key="nav_{page_path}"] {{
                        background: transparent !important;
                        border: none !important;
                        text-align: left !important;
                        font-weight: {'600' if is_active else '400'} !important;
                        padding: 0.6rem 1rem !important;
                        border-radius: 0 !important;
                        transition: all 0.2s ease !important;
                    }}
                    button[key="nav_{page_path}"]:hover {{
                        background: rgba(76, 175, 80, 0.1) !important;
                        color: #FFFFFF !important;
                    }}
                </style>
                """,
                unsafe_allow_html=True,
            )

        # ── Separator ──────────────────────────────────────────────
        st.markdown(
            "<hr style='margin: 1.5rem 0; border-color: rgba(255,255,255,0.1);'>",
            unsafe_allow_html=True,
        )

        # ── Theme Toggle ───────────────────────────────────────────
        from components.theme import render_theme_toggle
        render_theme_toggle()

        # ── Footer Info ────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="position: fixed; bottom: 1rem; left: 1rem; right: 1rem; 
                        text-align: center; font-size: 0.7rem; color: rgba(255,255,255,0.3);">
                Built with ❤️ using Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected_page


def navigate_to(page_path: str):
    """Set the current page in session state."""
    st.session_state["current_page"] = page_path


def get_current_page() -> str:
    """Get the currently active page path."""
    return st.session_state.get("current_page", "landing")


def init_navigation():
    """Initialize navigation state."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "landing"
