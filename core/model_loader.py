"""
Model loading utilities with caching for the Mushroom Classifier.
Handles loading of SVM model, label encoders, scaler, and image model.
"""
import json
import pickle

import streamlit as st

from config import (
    DATASET_PATH,
    FEATURE_ORDER,
    FEATURE_ORDER_PATH,
    FEATURE_VALUE_MAP,
    IMAGE_MODEL_PATH,
    LABEL_ENCODERS_PATH,
    SCALER_PATH,
    SVM_MODEL_PATH,
)
from core.data_transformer import transform_batch, ui_options_for_encoder


def load_feature_order(label_encoders: dict) -> list:
    """Load persisted training feature order, else use canonical CSV order."""
    if FEATURE_ORDER_PATH.exists():
        try:
            meta = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
            order = meta.get("feature_order") or []
            if order and all(f in label_encoders for f in order):
                return list(order)
        except Exception:
            pass
    return [f for f in FEATURE_ORDER if f in label_encoders]


@st.cache_resource(show_spinner="Loading prediction models...")
def load_manual_models():
    """
    Load the SVM model, label encoders, scaler, and authoritative feature order.

    Returns:
        tuple: (model, label_encoders, scaler, feature_options, feature_names)
    """
    try:
        for path, name in [
            (SVM_MODEL_PATH, "svm_model.pkl"),
            (LABEL_ENCODERS_PATH, "label_encoders.pkl"),
            (SCALER_PATH, "scaler.pkl"),
        ]:
            if not path.exists():
                st.error(
                    f"Model file not found: {name}. "
                    "Run `python train_svm.py` from the project root to create it."
                )
                return None, None, None, None, None

        with open(SVM_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(LABEL_ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        feature_names = load_feature_order(label_encoders)
        feature_options = {}
        for feature in feature_names:
            le = label_encoders[feature]
            if feature in FEATURE_VALUE_MAP:
                feature_options[feature] = ui_options_for_encoder(feature, le)
            else:
                feature_options[feature] = [str(c) for c in le.classes_]

        return model, label_encoders, scaler, feature_options, feature_names

    except pickle.UnpicklingError as e:
        st.error(f"Error loading model file (corrupted or incompatible format): {str(e)}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading manual prediction models: {str(e)}")
        return None, None, None, None, None


@st.cache_resource(show_spinner="Loading image classification model...")
def load_image_model():
    """Load image classification model from models/ directory."""
    from config import PYTORCH_IMAGE_MODEL_PATH, SKLEARN_IMAGE_MODEL_PATH, IMAGE_MODEL_PATH
    import pickle

    # 1. Try PyTorch model if PyTorch is installed
    if PYTORCH_IMAGE_MODEL_PATH.exists():
        try:
            import torch
            model = torch.jit.load(str(PYTORCH_IMAGE_MODEL_PATH))
            model.eval()
            return model
        except Exception:
            pass

    # 2. Try standalone Scikit-Learn image model (No PyTorch/TensorFlow required)
    if SKLEARN_IMAGE_MODEL_PATH.exists():
        try:
            with open(SKLEARN_IMAGE_MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception:
            pass

    # 3. Try Keras model if TensorFlow is available
    if IMAGE_MODEL_PATH.exists():
        try:
            from tensorflow.keras.models import load_model as keras_load_model
            return keras_load_model(str(IMAGE_MODEL_PATH))
        except Exception:
            pass

    return None


@st.cache_resource(show_spinner="Loading training data sample...")
def load_training_data_sample(n_samples: int = 50):
    """
    Load a scaled sample of the original training data for SHAP background.
    Encodes UCI letter codes with the same LabelEncoders used at train time.
    """
    try:
        import pandas as pd

        model, label_encoders, scaler, _feature_options, feature_names = load_manual_models()
        if model is None or scaler is None or not feature_names:
            return None

        if not DATASET_PATH.exists():
            return None

        df = pd.read_csv(DATASET_PATH)
        available = [f for f in feature_names if f in df.columns]
        if len(available) != len(feature_names):
            return None

        sample = df[available].sample(min(n_samples, len(df)), random_state=42)
        return transform_batch(sample, label_encoders, scaler, feature_names)

    except Exception:
        return None


@st.cache_data
def get_feature_options():
    """Get feature options without loading the full model."""
    try:
        if not LABEL_ENCODERS_PATH.exists():
            return {}

        with open(LABEL_ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)

        feature_names = load_feature_order(label_encoders)
        feature_options = {}
        for feature in feature_names:
            le = label_encoders[feature]
            if feature in FEATURE_VALUE_MAP:
                feature_options[feature] = ui_options_for_encoder(feature, le)
            else:
                feature_options[feature] = [str(c) for c in le.classes_]
        return feature_options
    except Exception:
        return {}


def get_feature_display_name(feature: str) -> str:
    from config import FEATURE_DISPLAY_NAMES
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace("-", " ").title())
