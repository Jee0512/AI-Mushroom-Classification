"""
Image preprocessing utilities for the Mushroom Classifier.
Prepares images for the EfficientNetB0 classification model.
"""
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess_input,
)


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Resize and normalize the input image for the EfficientNetB0 model.

    Args:
        image: PIL Image in RGB mode
        target_size: Target (height, width)

    Returns:
        numpy.ndarray: Preprocessed image array of shape (1, H, W, C)
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image
    image = image.resize(target_size)

    # Convert to numpy array
    img_array = np.array(image)

    # Expand dimensions to create a batch (1, H, W, C)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize using EfficientNet's specific preprocessing
    img_array = efficientnet_preprocess_input(img_array)

    return img_array


def get_image_preview(image: Image.Image, target_size=(224, 224)) -> Image.Image:
    """
    Return a resized preview of the image (for display purposes).

    Args:
        image: PIL Image
        target_size: Target size

    Returns:
        PIL Image resized to target size
    """
    return image.resize(target_size)


def validate_image(image: Image.Image) -> bool:
    """
    Validate that the uploaded image is usable.

    Args:
        image: PIL Image

    Returns:
        True if the image is valid, False otherwise
    """
    try:
        # Check image is not corrupt
        image.verify()
        # Re-open after verify (verify invalidates the image)
        return True
    except Exception:
        return False
