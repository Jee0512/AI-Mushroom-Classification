"""
Landing page - Hero section with project overview.
Professional landing page that introduces the application and its features.
"""
import streamlit as st
from config import APP_NAME, APP_TAGLINE, APP_VERSION, APP_DESCRIPTION, PAGES


def app():
    """Render the landing page."""
    
    # ── Hero Section ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="hero-section">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🍄</div>
            <h1 class="hero-title">{APP_NAME}</h1>
            <p class="hero-subtitle">{APP_TAGLINE}</p>
            <p style="font-size: 1rem; opacity: 0.85; max-width: 600px; margin: 0 auto; line-height: 1.6;">
                {APP_DESCRIPTION}
            </p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-value">22</div>
                    <div class="hero-stat-label">Features Analyzed</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">SVM</div>
                    <div class="hero-stat-label">ML Model</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">99%</div>
                    <div class="hero-stat-label">Accuracy</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">2</div>
                    <div class="hero-stat-label">Prediction Methods</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick Actions ─────────────────────────────────────────────
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div class="card" style="text-align: center; cursor: pointer;" onclick="alert('Navigate to Manual Prediction')">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔬</div>
                <div style="font-weight: 600;">Manual Prediction</div>
                <div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.25rem;">
                    Classify by features
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔬 Manual Prediction", key="quick_manual", use_container_width=True):
            st.session_state["current_page"] = "manual_prediction"
            st.rerun()
    
    with col2:
        st.markdown(
            """
            <div class="card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📸</div>
                <div style="font-weight: 600;">Image Prediction</div>
                <div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.25rem;">
                    Upload a photo
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📸 Image Prediction", key="quick_image", use_container_width=True):
            st.session_state["current_page"] = "image_prediction"
            st.rerun()
    
    with col3:
        st.markdown(
            """
            <div class="card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 600;">Analytics</div>
                <div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.25rem;">
                    View insights
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📊 Analytics", key="quick_analytics", use_container_width=True):
            st.session_state["current_page"] = "analytics"
            st.rerun()
    
    with col4:
        st.markdown(
            """
            <div class="card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📖</div>
                <div style="font-weight: 600;">Encyclopedia</div>
                <div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.25rem;">
                    Learn about features
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📖 Encyclopedia", key="quick_encyclopedia", use_container_width=True):
            st.session_state["current_page"] = "encyclopedia"
            st.rerun()

    st.markdown("---")

    # ── Features Overview ──────────────────────────────────────────
    st.markdown("## ✨ Key Features")
    
    features = [
        {
            "icon": "🔬",
            "title": "Manual Feature Prediction",
            "desc": "Select from 22 physical characteristics of a mushroom and get an instant edibility prediction with confidence scores and SHAP explainability.",
            "color": "#2E7D32",
        },
        {
            "icon": "📸",
            "title": "Image Recognition",
            "desc": "Upload a mushroom photo and let the deep learning model (EfficientNetB0) classify it. Grad-CAM visualizations show which regions influenced the decision.",
            "color": "#1565C0",
        },
        {
            "icon": "🤖",
            "title": "Explainable AI (XAI)",
            "desc": "Understand why the model made its prediction with SHAP force plots, waterfall charts, and natural language explanations for each feature.",
            "color": "#FF8F00",
        },
        {
            "icon": "📊",
            "title": "Analytics Dashboard",
            "desc": "Track prediction trends, view model performance metrics, explore dataset statistics, and generate comprehensive reports.",
            "color": "#7B1FA2",
        },
        {
            "icon": "📜",
            "title": "Prediction History",
            "desc": "All your predictions are saved locally. Search, filter, and export history as CSV or PDF for record-keeping.",
            "color": "#00838F",
        },
        {
            "icon": "📖",
            "title": "Mushroom Encyclopedia",
            "desc": "Learn about each mushroom feature, its values, and why it matters for classification. Educational content for beginners and experts alike.",
            "color": "#E65100",
        },
    ]
    
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="card" style="border-top: 3px solid {feature['color']}; height: 100%;">
                    <div class="card-header">
                        <span style="font-size: 1.5rem;">{feature['icon']}</span>
                        <span>{feature['title']}</span>
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                        {feature['desc']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── How It Works ───────────────────────────────────────────────
    st.markdown("## 🎯 How It Works")
    
    steps = [
        {
            "step": "1",
            "title": "Choose Your Method",
            "desc": "Select either Manual Prediction (feature-based) or Image Prediction (photo-based) depending on what information you have.",
            "icon": "1️⃣",
        },
        {
            "step": "2",
            "title": "Provide Input",
            "desc": "Fill in the mushroom's physical characteristics from dropdown menus, or upload a clear image of the mushroom.",
            "icon": "2️⃣",
        },
        {
            "step": "3",
            "title": "Get AI Analysis",
            "desc": "The SVM model (or EfficientNetB0 for images) analyzes the input and provides a prediction with confidence score.",
            "icon": "3️⃣",
        },
        {
            "step": "4",
            "title": "Understand the Decision",
            "desc": "Explore SHAP explanations, Grad-CAM heatmaps, and feature importance charts to understand why the AI made its prediction.",
            "icon": "4️⃣",
        },
    ]
    
    cols = st.columns(4)
    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(
                f"""
                <div class="card" style="text-align: center; height: 100%;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{step['icon']}</div>
                    <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">{step['title']}</div>
                    <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.5;">
                        {step['desc']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Tech Stack ─────────────────────────────────────────────────
    st.markdown("## 🛠️ Technology Stack")
    
    techs = [
        ("Streamlit", "Python Web Framework", "assets/streamlit.png"),
        ("Scikit-learn", "SVM Model", "assets/sklearn.png"),
        ("TensorFlow", "EfficientNetB0", "assets/tensorflow.png"),
        ("SHAP", "Explainable AI", "assets/shap.png"),
        ("ReportLab", "PDF Generation", "assets/reportlab.png"),
        ("Pandas", "Data Processing", "assets/pandas.png"),
    ]
    
    cols = st.columns(6)
    for i, (name, desc, _) in enumerate(techs):
        with cols[i]:
            st.markdown(
                f"""
                <div class="card" style="text-align: center;">
                    <div style="font-weight: 600; font-size: 0.95rem;">{name}</div>
                    <div style="font-size: 0.75rem; color: var(--text-light); margin-top: 0.25rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Footer Disclaimer ──────────────────────────────────────────
    st.markdown(
        """
        <div class="card" style="background: #FFF3E0; border-left: 4px solid #FF9800;">
            <div class="card-header">
                <span>⚠️</span>
                <span>Important Disclaimer</span>
            </div>
            <div style="color: #E65100; font-size: 0.9rem; line-height: 1.6;">
                <strong>This application is for educational and research purposes only.</strong> 
                The predictions made by this AI model should <strong>NOT</strong> be used as the sole 
                basis for determining if a mushroom is safe to eat. Many edible and poisonous mushrooms 
                share similar characteristics, and misidentification can have serious consequences. 
                Always consult with a qualified mycologist or mushroom expert before consuming any 
                wild mushroom.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
