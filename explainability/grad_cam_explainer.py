"""
Grad-CAM explainability utilities for the Mushroom Classifier.
Generates class activation heatmaps for image classification models.
"""
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None
from PIL import Image
import matplotlib.cm as cm


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.

    Args:
        img_array: Preprocessed image array (1, H, W, C)
        model: Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Optional predicted class index

    Returns:
        numpy.ndarray: Heatmap values normalized to [0, 1]
    """
    if tf is None:
        raise ImportError("TensorFlow is required for Grad-CAM but is not installed.")

    # Create a model that maps input to activations of the last conv layer and output
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the top predicted class wrt the feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Mean intensity of gradients over channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by importance and sum
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to 0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def find_last_conv_layer(model) -> str:
    """
    Find the last convolutional layer name in a Keras model.

    Args:
        model: Keras model

    Returns:
        str: Layer name, or None if no conv layer found
    """
    if keras is None:
        return None

    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D) and "conv" in layer.name:
            return layer.name
    return None


def generate_grad_cam_heatmap(model, img_array, pred_index=None, alpha=0.4):
    """
    Generate and overlay a Grad-CAM heatmap on the original image.

    Args:
        model: Keras model
        img_array: Preprocessed image array (1, H, W, C)
        pred_index: Optional predicted class index
        alpha: Overlay transparency (0-1)

    Returns:
        PIL.Image: Superimposed heatmap image
    """
    last_conv_layer_name = find_last_conv_layer(model)

    if not last_conv_layer_name:
        raise ValueError("Could not find a convolutional layer in the model for Grad-CAM.")

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[2]))
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose heatmap on original image
    original_img = keras.preprocessing.image.array_to_img(img_array[0])
    superimposed_img = Image.fromarray(
        np.uint8(original_img * (1 - alpha) + jet_heatmap * alpha)
    )

    return superimposed_img
