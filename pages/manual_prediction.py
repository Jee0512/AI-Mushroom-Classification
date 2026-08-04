"""
Manual Prediction Page - Enhanced with professional UI, confidence gauge,
probability charts, and improved SHAP explainability.
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from core.model_loader import load_manual_models, load_training_data_sample
from core.data_transformer import transform_input
from core.prediction_utils import calculate_confidence_and_risk, get_feature_explanations
from explainability.shap_explainer import get_shap_explanation, plot_shap_force, plot_shap_waterfall, plot_shap_bar
from components.cards import prediction_result_card, metric_card, feature_card, info_card
from components.charts import confidence_gauge, probability_bar_chart, feature_importance_chart
from config import FEATURE_DISPLAY_NAMES, FEATURE_DESCRIPTIONS, SUCCESS_COLOR, DANGER_COLOR


def app():
    """Render the Manual Prediction page."""
    
    # ── Page Header ────────────────────────────────────────────────
    st.markdown(
        """
        <div class="fade-in">
            <h1 style="margin-bottom: 0.5rem;">🔬 Manual Mushroom Classifier</h1>
            <p style="color: var(--text-secondary); font-size: 1.05rem; margin-bottom: 1.5rem;">
                Select the physical characteristics of a mushroom to predict whether it's edible or poisonous.
                Our AI model analyzes 22 different features with explainable AI insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load Models ────────────────────────────────────────────────
    with st.spinner("Loading prediction models..."):
        model, label_encoders, scaler, feature_options, feature_names = load_manual_models()

    if model is None:
        st.error("""
        ### Failed to load models
        Please ensure the following files exist in the `models/` directory:
        - `svm_model.pkl`
        - `label_encoders.pkl`
        - `scaler.pkl`
        """)
        return

    # ── Input Form ─────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📋 Mushroom Characteristics")
        st.markdown("Fill in the physical characteristics of the mushroom you want to classify.")
        
        # Create input dictionary
        input_dict = {}
        
        # Split features into two columns
        features = list(feature_options.keys())
        mid_point = len(features) // 2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Left Side**")
            for feature in features[:mid_point]:
                display_name = FEATURE_DISPLAY_NAMES.get(feature, feature.replace("-", " ").title())
                help_text = FEATURE_DESCRIPTIONS.get(feature, "")
                
                input_dict[feature] = st.selectbox(
                    f"{display_name}",
                    options=feature_options[feature],
                    key=f"manual_{feature}",
                    help=help_text,
                )
        
        with col2:
            st.markdown("**Right Side**")
            for feature in features[mid_point:]:
                display_name = FEATURE_DISPLAY_NAMES.get(feature, feature.replace("-", " ").title())
                help_text = FEATURE_DESCRIPTIONS.get(feature, "")
                
                input_dict[feature] = st.selectbox(
                    f"{display_name}",
                    options=feature_options[feature],
                    key=f"manual_{feature}",
                    help=help_text,
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Prediction Button ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔮 Predict Edibility", type="primary", use_container_width=True)
    
    with col_info:
        st.markdown(
            "<span style='color: var(--text-light); font-size: 0.85rem;'>"
            "The model will analyze 22 features and provide explainable AI insights.</span>",
            unsafe_allow_html=True,
        )

    # ── Prediction & Results ───────────────────────────────────────
    if predict_clicked:
        prediction_start_time = time.time()
        
        with st.spinner("Analyzing mushroom characteristics..."):
            # Transform input
            input_scaled = transform_input(input_dict, label_encoders, scaler)
            
            if input_scaled is not None:
                try:
                    # Get prediction probabilities
                    prediction_proba = model.predict_proba(input_scaled)[0]
                    prediction = np.argmax(prediction_proba)
                    
                    # 0 = Edible, 1 = Poisonous
                    predicted_class = "EDIBLE" if prediction == 0 else "POISONOUS"
                    confidence = prediction_proba[prediction] * 100
                    
                    prediction_end_time = time.time()
                    prediction_time = prediction_end_time - prediction_start_time
                    
                    # ── Results Section ────────────────────────────
                    st.markdown("---")
                    st.markdown("## 📊 Prediction Results")
                    
                    # Main prediction card
                    risk_level, risk_color, risk_icon = calculate_confidence_and_risk(confidence, predicted_class)
                    
                    col_result, col_gauge, col_probs = st.columns([1.5, 1, 1])
                    
                    with col_result:
                        prediction_result_card(predicted_class, confidence, risk_level)
                        
                        # Prediction metadata
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            metric_card("Model", "SVM", icon="🤖", color="#1565C0")
                        with col_m2:
                            metric_card("Response Time", f"{prediction_time:.2f}s", icon="⚡", color="#FF8F00")
                    
                    with col_gauge:
                        st.markdown("**Confidence Gauge**")
                        gauge_fig = confidence_gauge(confidence, predicted_class)
                        st.pyplot(gauge_fig)
                        plt.clf()
                    
                    with col_probs:
                        st.markdown("**Probability Distribution**")
                        prob_fig = probability_bar_chart(prediction_proba)
                        st.pyplot(prob_fig)
                        plt.clf()
                    
                    # ── Status Messages ────────────────────────────
                    if predicted_class == "EDIBLE":
                        st.success(f"🍄 This mushroom is predicted to be **{predicted_class}** with {confidence:.1f}% confidence.")
                    else:
                        st.error(f"☠️ This mushroom is predicted to be **{predicted_class}** with {confidence:.1f}% confidence.")
                        st.warning("⚠️ Never eat wild mushrooms without expert verification!")
                    
                    if confidence < 70:
                        st.warning("⚠️ Model confidence is low. Please consult a mushroom expert.")
                    
                    st.info(
                        "📌 **Note:** This is a prediction model and should not be used as the sole basis "
                        "for determining if a mushroom is safe to eat. Always consult with mycology experts."
                    )
                    
                    # ── Save to History ────────────────────────────
                    history_entry = {
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "method": "Manual",
                        "prediction": predicted_class,
                        "confidence": f"{confidence:.1f}%",
                        "risk_level": risk_level,
                        "features": input_dict.copy(),
                        "model": "Support Vector Machine (SVM)",
                    }
                    st.session_state["prediction_history"].append(history_entry)
                    
                    # ── SHAP Explainability ────────────────────────
                    st.markdown("---")
                    st.markdown("## 🤔 Why did the AI make this prediction?")
                    st.markdown(
                        "Explore the model's decision-making process with SHAP (SHapley Additive exPlanations). "
                        "These visualizations show how each feature contributed to the final prediction."
                    )
                    
                    try:
                        with st.spinner("Generating AI explanations..."):
                            # Load background data for SHAP
                            background_data = load_training_data_sample(n_samples=50)
                            
                            if background_data is None:
                                # Fallback: use a simple background
                                background_data = np.zeros((1, len(feature_names)))
                            
                            # Generate SHAP values
                            shap_values, expected_value = get_shap_explanation(
                                model, input_scaled, background_data, feature_names
                            )
                            
                            if shap_values is not None:
                                # ── Top Features ──────────────────
                                st.markdown("### 🔑 Top Contributing Features")
                                
                                # Get top features text
                                top_features_text, top_features_list = plot_shap_bar(
                                    shap_values, feature_names, display_streamlit=True
                                )
                                
                                st.info(top_features_text)
                                
                                # Feature importance chart
                                st.markdown("#### Feature Importance Chart")
                                shap_importance = shap_values[1][0]  # For poisonous class
                                importance_fig = feature_importance_chart(
                                    feature_names, shap_importance, top_n=10
                                )
                                st.pyplot(importance_fig)
                                plt.clf()
                                
                                # ── SHAP Visualizations ────────────
                                tab1, tab2, tab3 = st.tabs(
                                    ["📊 Force Plot", "🌊 Waterfall Plot", "📝 Detailed Explanation"]
                                )
                                
                                with tab1:
                                    st.markdown(
                                        "**Force Plot** — Shows how each feature pushes the prediction "
                                        "from the base value (average prediction) to the final output."
                                    )
                                    plot_shap_force(expected_value, shap_values, input_scaled, feature_names)
                                    st.pyplot(plt.gcf())
                                    plt.clf()
                                
                                with tab2:
                                    st.markdown(
                                        "**Waterfall Plot** — Illustrates how each feature's value "
                                        "moves the model output from the expected value to the current prediction."
                                    )
                                    plot_shap_waterfall(expected_value, shap_values, input_scaled, feature_names)
                                    st.pyplot(plt.gcf())
                                    plt.clf()
                                
                                with tab3:
                                    with st.expander("📝 Detailed Feature Explanations", expanded=True):
                                        st.markdown(
                                            "Here are beginner-friendly explanations of the key features "
                                            "that influenced this prediction:"
                                        )
                                        get_feature_explanations(input_dict, top_features_list)
                                
                                # ── Feature Cards ──────────────────
                                st.markdown("### 📋 Feature Breakdown")
                                st.markdown(
                                    "Each feature's selected value and its contribution to the prediction:"
                                )
                                
                                # Show feature cards in a grid
                                shap_values_for_class = shap_values[1][0]  # For poisonous class
                                feature_impacts = list(zip(feature_names, shap_values_for_class))
                                feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                                
                                cols = st.columns(2)
                                for i, (feat_name, impact) in enumerate(feature_impacts[:10]):
                                    with cols[i % 2]:
                                        feature_card(
                                            feat_name,
                                            input_dict.get(feat_name, "N/A"),
                                            importance=impact,
                                            description=FEATURE_DESCRIPTIONS.get(feat_name, ""),
                                        )
                    
                    except Exception as e:
                        st.error(f"Error generating AI explanations: {str(e)}")
                        st.info("Try using LIME as a fallback for local explanations if SHAP is not fully supported.")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("Failed to transform input data. Please check your selections.")
