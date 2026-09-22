import base64
import streamlit as st
from config import (
    FEATURE_DISPLAY_NAMES,
    FEATURE_GROUPS,
    FEATURE_MAPPING,
    FEATURE_ORDER,
    FEATURE_DESCRIPTIONS,
    GREEN_CENTER_PATH,
    SAFETY_DISCLAIMER,
)
from core.data_transformer import transform_input
from core.model_loader import load_manual_models
from core.prediction_utils import calculate_confidence_and_risk, predict_with_confidence
from utils.history_manager import save_prediction

st.set_page_config(page_title="Feature Classification | Mushroom Classifier", page_icon="🔬", layout="wide")
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

with st.spinner("Loading prediction models…"):
    model, label_encoders, scaler, feature_options, feature_names = load_manual_models()

if model is None:
    st.markdown("""
    <div class="info-card" style="margin-top:80px;">
        <span class="info-icon">⚠️</span>
        <div class="info-title">Models Not Found</div>
        <div class="info-body">Could not load SVM model files. Please ensure <code>svm_model.pkl</code>, <code>label_encoders.pkl</code> and <code>scaler.pkl</code> exist in <code>models/</code>.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

center_b64 = get_base64(GREEN_CENTER_PATH)
center_img_html = f'<img src="data:image/jpeg;base64,{center_b64}" style="width:220px;height:220px;border-radius:50%;border:3px solid rgba(16,185,129,0.35);box-shadow:0 0 40px rgba(16,185,129,0.2),0 12px 40px rgba(0,0,0,0.3);object-fit:cover;" alt="Mushroom">' if center_b64 else ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    res_class = result["class"]
    confidence = result["confidence"]
    is_edible = res_class == "EDIBLE"
    risk_level, _, _ = calculate_confidence_and_risk(confidence, res_class)

    card_cls = "result-edible" if is_edible else "result-poisonous"
    banner_cls = "result-edible" if is_edible else "result-poisonous"
    icon = "🍄" if is_edible else "☠️"
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
                <div class="result-meta-value">Support Vector Machine (SVM)</div>
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
        if st.button("🍄 Start New Prediction", use_container_width=True, type="primary"):
            st.session_state.prediction_result = None
            st.rerun()
    with res_col2:
        if st.button("⚡ Change Method", use_container_width=True):
            st.session_state.prediction_result = None
            st.switch_page("pages/1_select_method.py")
else:
    st.markdown(f"""
    <div class="page-header" style="margin-top:80px;">
        <span class="page-title">🍄 Mushroom Classification</h2>
        <p class="page-subtitle">Select the visible physical characteristics of your mushroom specimen below. Each trait helps our model determine edibility.</p>
    </div>
    """, unsafe_allow_html=True)

    # Centered mushroom image
    st.markdown(f"""
    <div style="display:flex;justify-content:center;align-items:center;margin-bottom:24px;">
        <div style="text-align:center;">
            {center_img_html}
            <p style="color:rgba(255,255,255,0.5);font-size:0.82rem;margin-top:10px;letter-spacing:0.5px;">Specimen Reference</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feat-layout" style="margin-top:16px;">
        <div class="feat-banner">
            <h3 class="feat-banner-title">Physical Characteristics</h3>
            <p class="feat-banner-sub">Choose an option for each characteristic from the dropdown menus.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("mushroom_classification_form"):
        input_dict = {}
        tabs = st.tabs(list(FEATURE_GROUPS.keys()))

        for tab, features_in_group in zip(tabs, FEATURE_GROUPS.values()):
            with tab:
                valid = [f for f in features_in_group if f in feature_names]
                half = (len(valid) + 1) // 2
                left = valid[:half]
                right = valid[half:]
                col_a, col_b, col_c = st.columns(3, gap="large")
                with col_a:
                    for feat in left:
                        label = FEATURE_DISPLAY_NAMES.get(feat, feat.replace("-", " ").title())
                        opts = feature_options.get(feat, [])
                        input_dict[feat] = st.selectbox(label, options=opts, key=f"f_{feat}")
                with col_b:
                    for feat in right:
                        label = FEATURE_DISPLAY_NAMES.get(feat, feat.replace("-", " ").title())
                        opts = feature_options.get(feat, [])
                        input_dict[feat] = st.selectbox(label, options=opts, key=f"f_{feat}")

        # ML Model Display
        st.markdown("---")
        model_col1, model_col2, model_col3, model_col4, model_col5 = st.columns(5)
        models_list = ["SVM (Default)", "Random Forest", "Neural Network", "KNN", "Naive Bayes"]
        for i, (col, mname) in enumerate(zip([model_col1, model_col2, model_col3, model_col4, model_col5], models_list)):
            with col:
                st.markdown(f"🧠 <div style='color:rgba(255,255,255,0.7);font-size:0.78rem;font-weight:600;text-align:center;line-height:1.4;'>{mname}</div>", unsafe_allow_html=True)
                if i == 0:
                    st.markdown("<div style='text-align:center;font-size:0.65rem;color:#6ee7b7;font-weight:700;'>● Active</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='text-align:center;font-size:0.65rem;color:rgba(255,255,255,0.3);'>● Inactive</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🍄 Predict Edibility", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("🧠 Analysing mushroom characteristics…"):
            try:
                input_scaled = transform_input(input_dict, label_encoders, scaler)
                prediction, prediction_proba = predict_with_confidence(model, input_scaled)

                if prediction is not None:
                    predicted_class = "EDIBLE" if prediction == 0 else "POISONOUS"
                    confidence = float(prediction_proba[prediction]) * 100
                    risk_level, _, _ = calculate_confidence_and_risk(confidence, predicted_class)

                    save_prediction({
                        "method": "Manual",
                        "prediction": predicted_class,
                        "confidence": f"{confidence:.1f}%",
                        "risk_level": risk_level,
                        "features": input_dict,
                        "model": "SVM",
                    })

                    st.session_state.prediction_result = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "features": input_dict,
                    }
                    st.rerun()
                else:
                    st.error("Prediction failed. Please try again.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Tips & Guide Cards
    st.markdown("---")
    st.markdown("""
    <div class="tip-grid">
        <div class="tip-card">
            <div class="tip-card-title">💡 Tips for Best Results</div>
            <div class="tip-card-body">Take clear photos in natural light. Select the most accurate option for each characteristic. When unsure, choose the most similar option.</div>
        </div>
        <div class="tip-card">
            <div class="tip-card-title">📖 Field Guide</div>
            <div class="tip-card-body">Compare your specimen with multiple reference images. Note the gill attachment and spore print color for reliable identification.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warn-box" style="margin-top:24px;">⚠️ <strong>Safety Warning:</strong> {SAFETY_DISCLAIMER}</div>
    """, unsafe_allow_html=True)
