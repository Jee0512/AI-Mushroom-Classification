"""
Model loading utilities with caching for the Mushroom Classifier.
Handles loading of SVM model, label encoders, scaler, and image model.
All paths are relative to the project root for portability.
"""
import streamlit as st
import pickle
import os
from pathlib import Path
from config import SVM_MODEL_PATH, LABEL_ENCODERS_PATH, SCALER_PATH, IMAGE_MODEL_PATH


@st.cache_resource(show_spinner="Loading prediction models...")
def load_manual_models():
    """
    Load the SVM model, label encoders, and scaler for manual prediction.
    Uses relative paths from the project root.
    Caches the resources to avoid reloading on every rerun.
    
    Returns:
        tuple: (model, label_encoders, scaler, feature_options, feature_names)
    """
    try:
        # Validate model files exist
        for path, name in [
            (SVM_MODEL_PATH, "svm_model.pkl"),
            (LABEL_ENCODERS_PATH, "label_encoders.pkl"),
            (SCALER_PATH, "scaler.pkl"),
        ]:
            if not path.exists():
                st.error(f"Model file not found: {name}. Please ensure it exists in the 'models' directory.")
                return None, None, None, None, None

        # Load models
        with open(SVM_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(LABEL_ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        # Build feature options from label encoders
        feature_options = {}
        feature_names = []
        for feature, le in label_encoders.items():
            feature_options[feature] = le.classes_.tolist()
            feature_names.append(feature)

        return model, label_encoders, scaler, feature_options, feature_names

    except pickle.UnpicklingError as e:
        st.error(f"Error loading model file (corrupted or incompatible format): {str(e)}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading manual prediction models: {str(e)}")
        return None, None, None, None, None


@st.cache_resource(show_spinner="Loading image classification model...")
def load_image_model():
    """
    Load the EfficientNetB0 model for image classification.
    Uses relative path from the project root.
    Caches the resource to avoid reloading on every rerun.
    
    Returns:
        tensorflow.keras.Model or None
    """
    if not IMAGE_MODEL_PATH.exists():
        st.error(f"Image model file not found: 'efficientnet_model.h5'. Please ensure it exists in the 'models' directory.")
        return None

    try:
        # Lazy import to avoid loading tensorflow if not needed
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(str(IMAGE_MODEL_PATH))
        return model
    except ImportError:
        st.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
        return None
    except Exception as e:
        st.error(f"Error loading image classification model: {str(e)}. Ensure the model file is valid.")
        return None


@st.cache_resource(show_spinner="Loading training data sample...")
def load_training_data_sample(n_samples: int = 100):
    """
    Load a sample of the original training data for SHAP background.
    This reads the CSV and transforms it using label encoders and scaler.
    
    Args:
        n_samples: Number of samples to load
    
    Returns:
        numpy.ndarray or None: Scaled training data sample
    """
    try:
        import pandas as pd
        from config import DATASET_PATH
        
        # Load models first to get encoders
        model, label_encoders, scaler, feature_options, feature_names = load_manual_models()
        if model is None or scaler is None:
            return None
        
        # Load dataset
        if not DATASET_PATH.exists():
            st.warning(f"Dataset file not found at {DATASET_PATH}. Using simplified SHAP background.")
            return None
        
        df = pd.read_csv(DATASET_PATH)
        
        # Drop the target column (usually 'class' or 'target')
        X = df.drop(columns=["class"], errors="ignore")
        
        # Ensure we have the right features
        available_features = [f for f in feature_names if f in X.columns]
        if len(available_features) < len(feature_names):
            st.warning("Some features missing from dataset. Using available features for SHAP background.")
        
        X = X[available_features]
        
        # Encode categorical features
        for feature in available_features:
            if feature in label_encoders:
                le = label_encoders[feature]
                X[feature] = X[feature].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Sample and scale
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)
        X_scaled = scaler.transform(X_sample)
        
        return X_scaled
    
    except Exception as e:
        st.warning(f"Could not load training data for SHAP background: {str(e)}")
        return None


@st.cache_data
def get_feature_options():
    """
    Get feature options without loading the full model.
    Useful for encyclopedia pages.
    
    Returns:
        dict: Feature name -> list of options
    """
    try:
        if not LABEL_ENCODERS_PATH.exists():
            return {}
        
        with open(LABEL_ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
        
        feature_options = {}
        for feature, le in label_encoders.items():
            feature_options[feature] = le.classes_.tolist()
        
        return feature_options
    except Exception:
        return {}


def get_feature_display_name(feature: str) -> str:
    """Get the human-readable display name for a feature."""
    from config import FEATURE_DISPLAY_NAMES
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace("-", " ").title())
