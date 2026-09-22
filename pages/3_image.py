import streamlit as st
from config import SAFETY_DISCLAIMER

st.set_page_config(page_title="Image Classification | Mushroom Classifier", page_icon="📸", layout="wide")
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
    <div class="nav-badge">Vision AI</div>
</div>
""", unsafe_allow_html=True)

from core.model_loader import load_image_model
from core.prediction_utils import calculate_confidence_and_risk
from utils.history_manager import save_prediction
from vision.image_classifier import predict_image_class
from vision.image_processor import preprocess_image

image_model = load_image_model()

if image_model is None:
    st.markdown("""
    <div class="info-card" style="margin-top:80px;">
        <span class="info-icon">⚠️</span>
        <div class="info-title">Image Model Unavailable</div>
        <div class="info-body">Please ensure model files exist in <code>models/</code>. Supported: <code>image_model.pt</code> (PyTorch) or <code>image_model.pkl</code> (Scikit-Learn).</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if "img_result" not in st.session_state:
    st.session_state.img_result = None
if "img_uploaded" not in st.session_state:
    st.session_state.img_uploaded = False

st.markdown("""
<div class="page-header" style="margin-top:80px;">
    <span class="page-title">📸 Image Classification</span>
    <p class="page-subtitle">Upload a clear photograph of your mushroom specimen for instant deep learning analysis.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.img_result:
    result = st.session_state.img_result
    res_class = result["class"]
    confidence = result["confidence"]
    is_edible = res_class == "EDIBLE"
    risk_level, _, _ = calculate_confidence_and_risk(confidence, res_class)
    icon = "🍄" if is_edible else "☠️"
    card_cls = "result-edible" if is_edible else "result-poisonous"
    banner_cls = "result-edible" if is_edible else "result-poisonous"
    badge_cls = "result-badge-safe" if risk_level in ("Safe", "Low Risk") else ("result-badge-unc" if "Uncertain" in risk_level else "result-badge-risk")

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <span class="result-big-icon">{icon}</span>
        <div class="result-banner {banner_cls}">{res_class}</div>
        <span class="result-badge {badge_cls}">{risk_level}</span>
        <div class="result-meta">
            <div class="result-meta-item">
                <div class="result-meta-label">Confidence</div>
                <div class="result-meta-value">{confidence:.1f}%</div>
            </div>
            <div class="result-meta-item">
                <div class="result-meta-label">Model</div>
                <div class="result-meta-value">Vision Classifier</div>
            </div>
            <div class="result-meta-item">
                <div class="result-meta-label">Risk Level</div>
                <div class="result-meta-value">{risk_level}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        if st.button("📷 Analyze Another Image", use_container_width=True, type="primary"):
            st.session_state.img_result = None
            st.session_state.img_uploaded = False
            st.rerun()
    with res_col2:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.img_result = None
            st.session_state.img_uploaded = False
            st.switch_page("pages/0_home.py")
else:
    st.markdown("""
    <div class="upload-zone">
        <span class="upload-big-icon">📷</span>
        <div class="upload-title">Drag & Drop Your Mushroom Image</div>
        <div class="upload-hint">Supported formats: JPG, JPEG, PNG · Max file size: 10 MB</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Browse Files", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="img_up")

    if uploaded is not None:
        st.session_state.img_uploaded = True
        from PIL import Image as PILImage
        image = PILImage.open(uploaded).convert("RGB")

        img_col1, img_col2 = st.columns([2, 1])
        with img_col1:
            st.image(image, caption="Uploaded Specimen Preview")

        analyze_col1, analyze_col2 = st.columns([1, 1])
        with analyze_col1:
            if st.button("🍄 Identify Mushroom", use_container_width=True, type="primary"):
                with st.spinner("🔬 Analyzing mushroom image features…"):
                    try:
                        processed = preprocess_image(image)
                        proba, idx = predict_image_class(image_model, processed)
                        if proba is not None:
                            pc = "EDIBLE" if idx == 0 else "POISONOUS"
                            conf = float(proba[idx]) * 100
                            risk, _, _ = calculate_confidence_and_risk(conf, pc)
                            save_prediction({
                                "method": "Image",
                                "prediction": pc,
                                "confidence": f"{conf:.1f}%",
                                "risk_level": risk,
                                "image_name": uploaded.name,
                                "model": "Vision Classifier",
                            })
                            st.session_state.img_result = {"class": pc, "confidence": conf}
                            st.rerun()
                        else:
                            st.error("Failed to process image. Try another photograph.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        with analyze_col2:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.img_result = None
                st.session_state.img_uploaded = False
                st.switch_page("pages/0_home.py")

    if not st.session_state.img_uploaded:
        tips_col1, tips_col2 = st.columns(2)
        with tips_col1:
            st.markdown("""
            <div class="tip-card">
                <div class="tip-card-title">💡 Photography Tips</div>
                <div class="tip-card-body">Use well-lit, close-up photos showing the cap, gills, and stalk clearly. Avoid blurry or dark images.</div>
            </div>
            """, unsafe_allow_html=True)
        with tips_col2:
            st.markdown("""
            <div class="tip-card">
                <div class="tip-card-title">📖 Image Guide</div>
                <div class="tip-card-body">Capture the mushroom from multiple angles. Ensure the cap and gills are fully visible for accurate analysis.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warn-box" style="margin-top:24px;">⚠️ <strong>Safety Warning:</strong> {SAFETY_DISCLAIMER}</div>
    """, unsafe_allow_html=True)
