"""
Settings Page - Application configuration and preferences.
"""
import streamlit as st
from components.theme import get_theme, is_dark
from config import APP_VERSION, APP_NAME


def app():
    """Render the Settings page."""
    
    st.markdown(
        """
        <div class="fade-in">
            <h1 style="margin-bottom: 0.5rem;">⚙️ Settings</h1>
            <p style="color: var(--text-secondary); font-size: 1.05rem; margin-bottom: 1.5rem;">
                Configure application preferences and view system information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Theme Settings ─────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎨 Appearance")
    
    col_t1, col_t2 = st.columns([1, 3])
    
    with col_t1:
        current_theme = get_theme()
        theme_icon = "🌙" if current_theme == "light" else "☀️"
        theme_label = "Dark Mode" if current_theme == "light" else "Light Mode"
        
        if st.button(f"{theme_icon} Toggle {theme_label}", use_container_width=True):
            from components.theme import toggle_theme
            toggle_theme()
            st.rerun()
    
    with col_t2:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0;">
                <p><strong>Current Theme:</strong> {'🌙 Dark' if is_dark() else '☀️ Light'}</p>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">
                    Toggle between light and dark mode for comfortable viewing in different environments.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Prediction Settings ────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Prediction Preferences")
    
    # Default prediction method
    default_method = st.radio(
        "Default Prediction Method",
        options=["Manual", "Image"],
        index=0,
        help="Select which prediction page to show by default when you click 'Quick Predict'.",
        horizontal=True,
    )
    st.session_state["default_method"] = default_method
    
    # Show SHAP explanations
    show_shap = st.toggle(
        "Show SHAP Explanations",
        value=st.session_state.get("show_shap", True),
        help="Enable or disable SHAP feature importance visualizations after predictions.",
    )
    st.session_state["show_shap"] = show_shap
    
    # Confidence threshold
    st.markdown("#### Confidence Threshold")
    st.markdown(
        "Set the minimum confidence threshold for predictions. Predictions below this threshold "
        "will show a warning."
    )
    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=50,
        max_value=99,
        value=st.session_state.get("confidence_threshold", 70),
        step=5,
        help="Predictions below this confidence will trigger a warning.",
    )
    st.session_state["confidence_threshold"] = confidence_threshold
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Data Management ────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💾 Data Management")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        history_count = len(st.session_state.get("prediction_history", []))
        st.markdown(f"**Prediction History Records:** {history_count}")
        
        if history_count > 0:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["prediction_history"] = []
                st.success("✅ Prediction history cleared.")
                st.rerun()
    
    with col_d2:
        st.markdown("**Session State Variables:**")
        session_vars = [k for k in st.session_state.keys() if not k.startswith("_")]
        st.code("\n".join(session_vars))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── System Information ─────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🖥️ System Information")
    
    import sys
    import platform
    
    sys_info = [
        ("Application", f"{APP_NAME} v{APP_VERSION}"),
        ("Python Version", sys.version.split()[0]),
        ("Platform", platform.platform()),
        ("Streamlit Version", st.__version__),
        ("Session State Size", f"{len(st.session_state)} variables"),
    ]
    
    for label, value in sys_info:
        st.markdown(f"- **{label}:** `{value}`")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── About ──────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📝 About")
    
    st.markdown(
        f"""
        <p>
            <strong>{APP_NAME}</strong> is a production-grade machine learning application 
            for classifying mushroom edibility. It uses a Support Vector Machine (SVM) model 
            trained on the UCI Mushroom Dataset with 22 features and 8,124 samples.
        </p>
        <p>
            The application also features image-based classification using EfficientNetB0 
            transfer learning, explainable AI with SHAP and Grad-CAM, and comprehensive 
            analytics and reporting capabilities.
        </p>
        <p style="color: var(--text-light); font-size: 0.85rem;">
            Built with ❤️ using Streamlit, Scikit-learn, TensorFlow, and SHAP.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
