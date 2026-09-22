import streamlit as st
from config import SAFETY_DISCLAIMER

st.set_page_config(page_title="About | Mushroom Classifier", page_icon="ℹ️", layout="wide")
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
    <div class="nav-badge">About</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header" style="margin-top:80px;">
    <span class="page-title">ℹ️ About this Project</span>
    <p class="page-subtitle">Information about the Mushroom Classifier</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card" style="padding:36px;margin-bottom:24px;">
    <h3 style="color:#fff;font-family:'Playfair Display',serif;font-size:1.4rem;margin-bottom:12px;">Model Information</h3>
    <p style="color:rgba(255,255,255,0.75);line-height:1.7;">This application uses a Support Vector Machine (SVM) model trained on the UCI Mushroom Dataset. It evaluates 19 physical characteristics to classify a mushroom as either <strong style="color:#6ee7b7;">Edible</strong> or <strong style="color:#f87171;">Poisonous</strong>.</p>
    <p style="color:rgba(255,255,255,0.75);line-height:1.7;margin-top:12px;">The image classification feature uses a <strong style="color:#6ee7b7;">MobileNetV2</strong> deep learning model trained specifically for mushroom edibility detection.</p>
</div>

<div class="glass-card" style="padding:36px;margin-bottom:24px;border-left:4px solid #fbbf24;">
    <h3 style="color:#fff;font-family:'Playfair Display',serif;font-size:1.4rem;margin-bottom:12px;">⚠️ Disclaimer</h3>
    <p style="color:rgba(255,255,255,0.75);line-height:1.7;"><strong style="color:#fbbf24;">This tool is for educational purposes only.</strong> Never consume wild mushrooms based on an AI prediction. The classifications provided by this application may not be 100% accurate. Always consult an expert mycologist before consuming wild mushrooms.</p>
</div>

<div class="warn-box" style="margin-top:20px;">⚠️ <strong>Safety Warning:</strong> {SAFETY_DISCLAIMER}</div>

<div class="glass-card" style="padding:36px;margin-bottom:24px;">
    <h3 style="color:#fff;font-family:'Playfair Display',serif;font-size:1.4rem;margin-bottom:16px;">Features</h3>
    <ul style="color:rgba(255,255,255,0.75);line-height:2;">
        <li><strong>Manual Prediction:</strong> Enter physical traits to get an instant prediction.</li>
        <li><strong>Image Prediction:</strong> Upload an image for deep-learning-based classification (requires PyTorch model).</li>
        <li><strong>History & Analytics:</strong> Review past predictions and check statistics.</li>
    </ul>
</div>

<div class="glass-card" style="padding:36px;">
    <h3 style="color:#fff;font-family:'Playfair Display',serif;font-size:1.4rem;margin-bottom:16px;">Technical Details</h3>
    <ul style="color:rgba(255,255,255,0.75);line-height:2;">
        <li><strong>Frontend:</strong> Streamlit</li>
        <li><strong>ML Framework:</strong> Scikit-Learn, PyTorch</li>
        <li><strong>Data:</strong> UCI Machine Learning Repository</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:24px;">
    <button class="btn-secondary" onclick="window.location.href='/'">🏠 Back to Home</button>
</div>
""", unsafe_allow_html=True)
