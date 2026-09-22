import streamlit as st
from config import SAFETY_DISCLAIMER

st.set_page_config(page_title="Select Method | Mushroom Classifier", page_icon="⚡", layout="wide")
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
    <div class="nav-badge">Select Method</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header" style="margin-top:80px;">
    <h2 class="page-title">How Would You Like to Classify?</h2>
    <p class="page-subtitle">Choose a method to analyze your mushroom specimen.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="method-grid">
    <div class="method-card">
        <div class="method-icon">🔬</div>
        <div class="method-title">Feature Classification</div>
        <div class="method-desc">Select physical mushroom features such as cap shape, gill color, stalk root, odor, and 18 more characteristics using clear human-readable dropdown options. Our SVM model will determine if it's edible or poisonous.</div>
        <button class="btn-primary" id="feat-btn" onclick="document.querySelector('.stButton>button[data-testid=baseButton-primary']').click()">Feature Classification</button>
    </div>
    <div class="method-card">
        <div class="method-icon">📷</div>
        <div class="method-title">Image Classification</div>
        <div class="method-desc">Upload a clear photograph of a mushroom specimen. Our deep learning visual model will analyze the image and classify edibility instantly. Supports JPG, JPEG, and PNG formats up to 10MB.</div>
        <button class="btn-primary" id="img-btn" onclick="document.querySelector('.stButton>button[data-testid=baseButton-secondary]').click()">Image Classification</button>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🔬 Feature Classification", use_container_width=True, type="primary"):
        st.switch_page("pages/2_manual.py")
with col2:
    if st.button("📷 Image Classification", use_container_width=True, type="primary"):
        st.switch_page("pages/3_image.py")

st.markdown(f"""
<div class="warn-box" style="margin-top:32px;">⚠️ <strong>Safety Warning:</strong> {SAFETY_DISCLAIMER}</div>
""", unsafe_allow_html=True)
