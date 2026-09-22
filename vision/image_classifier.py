"""
Image classification utilities for the Mushroom Classifier.
Makes predictions using PyTorch, Scikit-Learn, or Keras image classification models.
"""
import numpy as np

def predict_image_class(model, processed_data):
    """
    Make a prediction using the loaded image classification model.

    Args:
        model: Loaded PyTorch, Scikit-Learn, or Keras model
        processed_data: Payload dictionary or array from preprocess_image

    Returns:
        Tuple of (prediction_proba, predicted_class_idx), or (None, None) on error
    """
    if model is None:
        return None, None

    if isinstance(processed_data, dict):
        tensor = processed_data.get("tensor")
        array = processed_data.get("array")
        features = processed_data.get("features")
    else:
        tensor = processed_data
        array = processed_data
        features = None

    # 1. Try PyTorch model inference if tensor and PyTorch available
    if tensor is not None:
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                with torch.no_grad():
                    output = model(tensor)
                    proba = output[0].cpu().numpy()
                    idx = int(np.argmax(proba))
                    return proba, idx
        except Exception:
            pass

    # 2. Try Scikit-Learn vision model inference
    if hasattr(model, "predict_proba"):
        try:
            if features is None and array is not None:
                from build_sklearn_image_model import extract_image_features
                features = extract_image_features(array)

            if features is not None:
                proba = model.predict_proba([features])[0]
                idx = int(np.argmax(proba))
                return proba, idx
        except Exception as e:
            print(f"Error during Scikit-Learn image prediction: {e}")

    # 3. Try Keras model inference
    if array is not None and hasattr(model, "predict"):
        try:
            norm_array = np.expand_dims(array, axis=0) if len(array.shape) == 3 else array
            predictions = model.predict(norm_array, verbose=0)
            proba = predictions[0]
            idx = int(np.argmax(proba))
            return proba, idx
        except Exception as e:
            print(f"Error during Keras image prediction: {e}")

    return None, None
