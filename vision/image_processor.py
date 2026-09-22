"""
Image preprocessing utilities for the Mushroom Classifier.
Prepares images for PyTorch, Scikit-Learn, or Keras image classification models.
"""
import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Resize and normalize the input PIL Image into a multi-format payload for models.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = image.resize(target_size)
    arr = np.array(resized, dtype=np.uint8)
    try:
        from build_sklearn_image_model import extract_image_features
        feats = extract_image_features(arr)
    except Exception:
        feats = None

    tensor = None
    if TORCH_AVAILABLE:
        try:
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            tensor = transform(image).unsqueeze(0)
        except Exception:
            tensor = None

    return {
        "tensor": tensor,
        "array": arr,
        "features": feats,
    }


def get_image_preview(image: Image.Image, target_size=(224, 224)) -> Image.Image:
    """Return a resized preview of the image."""
    return image.resize(target_size)


def validate_image(image: Image.Image) -> bool:
    """Validate that the uploaded image is usable."""
    try:
        image.verify()
        return True
    except Exception:
        return False
