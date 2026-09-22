"""
SHAP explainability utilities for the Mushroom Classifier.
KernelExplainer is used because it works with any sklearn estimator.

SHAP is optional. Failures must never break the main prediction flow.
"""
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    shap = None


def _model_predict_proba_wrapper(model):
    """Batch predict_proba when available; otherwise class-one-hot from predict()."""

    def predict_fn(X):
        X = np.atleast_2d(X)
        if hasattr(model, "predict_proba"):
            try:
                return np.asarray(model.predict_proba(X), dtype=float)
            except Exception:
                pass
        preds = model.predict(X)
        out = np.zeros((len(preds), 2), dtype=float)
        for i, p in enumerate(preds):
            idx = int(p)
            if 0 <= idx < 2:
                out[i, idx] = 1.0
            else:
                out[i, 1] = 1.0
        return out

    return predict_fn


def extract_class_shap(shap_values, class_idx: int = 1) -> np.ndarray | None:
    """Return a 1-D SHAP vector for one class from KernelExplainer output."""
    try:
        if shap_values is None:
            return None
        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[min(class_idx, len(shap_values) - 1)])
            return arr[0] if arr.ndim == 2 else arr.ravel()
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            c = min(class_idx, arr.shape[2] - 1)
            return arr[0, :, c]
        if arr.ndim == 2:
            return arr[0]
        if arr.ndim == 1:
            return arr
    except Exception:
        return None
    return None


def get_shap_explanation(model, input_scaled, background_data, feature_names):
    """
    Generate SHAP values for a given prediction.

    Returns:
        Tuple of (shap_values, expected_value), or (None, None) on error
    """
    if not SHAP_AVAILABLE or shap is None:
        return None, None

    try:
        n_features = len(feature_names)
        if background_data is None:
            background_data = np.zeros((10, n_features))
        background_data = np.atleast_2d(background_data)

        max_background_samples = 30
        if len(background_data) > max_background_samples:
            indices = np.linspace(
                0, len(background_data) - 1, max_background_samples, dtype=int
            )
            background_data = background_data[indices]

        explainer = shap.KernelExplainer(
            _model_predict_proba_wrapper(model), background_data
        )
        shap_values = explainer.shap_values(np.atleast_2d(input_scaled), nsamples=64)
        expected_value = explainer.expected_value
        return shap_values, expected_value
    except Exception:
        return None, None


def plot_shap_force(expected_value, shap_values, input_scaled, feature_names):
    if not SHAP_AVAILABLE or shap is None or plt is None:
        return None
    try:
        values = extract_class_shap(shap_values, class_idx=1)
        if values is None:
            return None
        if isinstance(expected_value, (list, np.ndarray)):
            expected_val = np.ravel(expected_value)
            expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]
        else:
            expected_val = expected_value
        shap.initjs()
        return shap.force_plot(
            expected_val,
            values,
            np.atleast_2d(input_scaled)[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
    except Exception:
        return None


def plot_shap_waterfall(expected_value, shap_values, input_scaled, feature_names):
    if not SHAP_AVAILABLE or shap is None or plt is None:
        return None
    try:
        values = extract_class_shap(shap_values, class_idx=1)
        if values is None:
            return None
        if isinstance(expected_value, (list, np.ndarray)):
            expected_val = np.ravel(expected_value)
            expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]
        else:
            expected_val = expected_value
        explanation = shap.Explanation(
            values=values,
            base_values=expected_val,
            data=np.atleast_2d(input_scaled)[0],
            feature_names=feature_names,
        )
        shap.waterfall_plot(explanation, show=False)
        return plt.gcf()
    except Exception:
        return None


def plot_shap_bar(shap_values, feature_names, display_streamlit=False, predicted_class=None):
    """
    Rank features by |SHAP| for the poisonous class.

    If display_streamlit: (summary_text, top_features_list)
    Otherwise: list of (feature_name, signed_shap) sorted by abs impact.
    """
    try:
        values = extract_class_shap(shap_values, class_idx=1)
        if values is None:
            return ("", []) if display_streamlit else []

        impact = np.abs(values)
        sorted_indices = np.argsort(impact)[::-1]
        ranked = [
            (feature_names[i], float(values[i]))
            for i in sorted_indices
            if i < len(feature_names)
        ]

        if display_streamlit:
            top_five = [name for name, _ in ranked[:5]]
            if top_five:
                pretty = ", ".join(f.replace("-", " ") for f in top_five)
                cls = (predicted_class or "the predicted class").lower()
                summary = (
                    f"The features that most influenced the model (toward poisonous vs edible) "
                    f"were {pretty}. The predicted class was {cls}."
                )
            else:
                summary = "The model's decision was influenced by a combination of features."
            return summary, top_five

        return ranked
    except Exception:
        return ("", []) if display_streamlit else []
