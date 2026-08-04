"""
Image classification utilities for the Mushroom Classifier.
Makes predictions using the loaded EfficientNetB0 model.
"""
import numpy as np
from tensorflow.keras.models import Model


def predict_image_class(model: Model, processed_image: np.ndarray):
    """
    Make a prediction using the loaded image classification model.

    Args:
        model: Loaded Keras model
        processed_image: Preprocessed image array (1, H, W, C)

    Returns:
        Tuple of (prediction_proba, predicted_class_idx), or (None, None) on error
    """
    try:
        # Model is expected to output probabilities for 2 classes (edible, poisonous)
        predictions = model.predict(processed_image, verbose=0)

        # Get the probabilities for each class
        prediction_proba = predictions[0]

        # Get the index of the class with the highest probability
        predicted_class_idx = np.argmax(prediction_proba)

        return prediction_proba, int(predicted_class_idx)

    except Exception as e:
        print(f"Error during image classification prediction: {str(e)}")
        return None, None
