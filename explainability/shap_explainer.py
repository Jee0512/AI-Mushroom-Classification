"""
SHAP explainability utilities for the Mushroom Classifier.
Generates model-agnostic explanations using KernelExplainer.
"""
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np


def get_shap_explanation(model, input_scaled, background_data, feature_names):
    """
    Generate SHAP values for a given prediction.

    Args:
        model: Trained sklearn model with predict_proba
        input_scaled: Scaled input array (1, n_features)
        background_data: Representative scaled training data sample
        feature_names: List of feature names

    Returns:
        Tuple of (shap_values, expected_value), or (None, None) on error
    """
    try:
        # Ensure background data is a 2D array
        if background_data is None:
            background_data = np.zeros((1, len(feature_names)))

        if getattr(background_data, "ndim", 0) == 1:
            background_data = background_data.reshape(1, -1)

        # Use a small background sample to keep KernelExplainer fast
        max_background_samples = 50
        if len(background_data) > max_background_samples:
            # Subsample deterministically
            indices = np.linspace(
                0, len(background_data) - 1, max_background_samples, dtype=int
            )
            background_data = background_data[indices]

        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        shap_values = explainer.shap_values(input_scaled, nsamples=100)
        expected_value = explainer.expected_value

        return shap_values, expected_value

    except Exception as e:
        st.error(f"SHAP explanation error: {str(e)}")
        return None, None


def plot_shap_force(expected_value, shap_values, input_scaled, feature_names):
    """
    Render a SHAP force plot.

    Args:
        expected_value: Base value(s) from the explainer
        shap_values: SHAP values per class
        input_scaled: Scaled input (1, n_features)
        feature_names: List of feature names
    """
    try:
        # For binary classification, explain the 'poisonous' class (index 1)
        class_idx = 1 if len(shap_values) > 1 else 0

        shap.initjs()

        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
            expected_val = expected_value[class_idx]
        else:
            expected_val = expected_value

        force_plot = shap.force_plot(
            expected_val,
            shap_values[class_idx][0],
            input_scaled[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        return force_plot

    except Exception as e:
        st.warning(f"Could not generate SHAP force plot: {str(e)}")
        return None


def plot_shap_waterfall(expected_value, shap_values, input_scaled, feature_names):
    """
    Render a SHAP waterfall plot.

    Args:
        expected_value: Base value(s) from the explainer
        shap_values: SHAP values per class
        input_scaled: Scaled input (1, n_features)
        feature_names: List of feature names
    """
    try:
        class_idx = 1 if len(shap_values) > 1 else 0

        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
            expected_val = expected_value[class_idx]
        else:
            expected_val = expected_value

        explanation = shap.Explanation(
            values=shap_values[class_idx][0],
            base_values=expected_val,
            data=input_scaled[0],
            feature_names=feature_names,
        )

        shap.waterfall_plot(explanation, show=False)
        return plt.gcf()

    except Exception as e:
        st.warning(f"Could not generate SHAP waterfall plot: {str(e)}")
        return None


def plot_shap_bar(shap_values, feature_names, display_streamlit=False):
    """
    Process SHAP bar chart data and optionally return a natural-language summary.

    Args:
        shap_values: SHAP values per class
        feature_names: List of feature names
        display_streamlit: If True, return (text_summary, top_features_list)

    Returns:
        If display_streamlit: (summary_text, top_features_list)
        Otherwise: List of (feature_name, impact) tuples
    """
    try:
        # For binary classification, use the 'poisonous' class (index 1)
        class_idx = 1 if len(shap_values) > 1 else 0
        feature_impact = np.abs(shap_values[class_idx][0])
        sorted_indices = np.argsort(feature_impact)[::-1]

        top_features = [
            (feature_names[i], feature_impact[i])
            for i in sorted_indices
            if feature_impact[i] > 0
        ]

        if display_streamlit:
            top_five = [f for f, _ in top_features[:5]]
            if top_five:
                summary = (
                    "The mushroom was classified as poisonous mainly because of its "
                    + ", ".join(f.replace("-", " ") for f in top_five[:5])
                    + "."
                )
            else:
                summary = "The model's decision was influenced by a combination of features."
            return summary, top_five

        return top_features

    except Exception as e:
        st.warning(f"Error processing SHAP bar chart data: {str(e)}")
        return "", [] if display_streamlit else []
