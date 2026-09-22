import base64
import streamlit as st
from config import ORANGE_HERO_PATH, SAFETY_DISCLAIMER

st.set_page_config(page_title="Mushroom Classifier | AI Edibility Analysis", page_icon="🍄", layout="wide")

st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@600;700;800;900&display=swap');</style>", unsafe_allow_html=True)
st.markdown(f"<style>{open('assets/styles.css', 'r', encoding='utf-8').read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="nav-header">
    <div class="nav-brand">
        <span class="nav-logo">🍄</span>
        <div class="nav-title-group">
            <div class="nav-title">Mushroom Identifier</div>
            <div class="nav-subtitle">AI Edibility Analysis</div>
        </div>
    </div>
    <div class="nav-badge">SVM Classifier v2.0</div>
</div>
""", unsafe_allow_html=True)


def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

hero_b64 = get_base64(ORANGE_HERO_PATH)
bg = f"background-image: url('data:image/jpeg;base64,{hero_b64}'); background-size: cover; background-position: center;" if hero_b64 else "background: linear-gradient(145deg, #143d28, #0a1f14);"

st.markdown(f"""
<div class="hero-section" style="{bg}">
    <div class="hero-content">
        <span class="hero-badge">🍄 Intelligent Mycology Platform</span>
        <h1 class="hero-title">Identify Mushrooms Instantly</h1>
        <p class="hero-subtitle">Classify wild mushrooms as edible or poisonous using physical characteristics or photograph analysis powered by machine learning.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="highlight-grid">
    <div class="highlight-card">
        <span class="highlight-icon">🔬</span>
        <div class="highlight-title">Feature Analysis</div>
        <div class="highlight-desc">22 physical traits with clear dropdown options</div>
    </div>
    <div class="highlight-card">
        <span class="highlight-icon">📸</span>
        <div class="highlight-title">Vision AI</div>
        <div class="highlight-desc">Deep learning analysis of mushroom photos</div>
    </div>
    <div class="highlight-card">
        <span class="highlight-icon">⚡</span>
        <div class="highlight-title">SVM Intelligence</div>
        <div class="highlight-desc">Precision model trained on UCI dataset</div>
    </div>
    <div class="highlight-card">
        <span class="highlight-icon">🛡️</span>
        <div class="highlight-title">Safety System</div>
        <div class="highlight-desc">Confidence scores and expert warnings</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="cta-primary">', unsafe_allow_html=True)
    if st.button("Start Analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/1_select_method.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="warn-box">⚠️ <strong>Safety Warning:</strong> {SAFETY_DISCLAIMER}</div>
""", unsafe_allow_html=True)
