"""
Prediction utilities for the Mushroom Classifier.
Handles risk calculation, confidence evaluation, and feature explanations.
"""
import streamlit as st
import numpy as np
from typing import Tuple, Dict, List


def predict_with_scores(model, input_scaled: np.ndarray) -> dict:
    """
    Predict class and return an honest score payload.

    Never fabricates probabilities. If predict_proba is available it is used
    and labeled as a model probability (Platt-scaled for SVM). Otherwise a
    raw decision_function score is returned and must not be shown as a %.
    """
    prediction = model.predict(input_scaled)[0]
    predicted_class_idx = int(prediction)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(input_scaled)[0], dtype=float)
        except Exception:
            proba = None

    decision_score = None
    if hasattr(model, "decision_function"):
        try:
            decision_score = float(np.ravel(model.decision_function(input_scaled))[0])
        except Exception:
            decision_score = None

    if proba is not None:
        score_type = "probability"
        score_value = float(proba[predicted_class_idx])
        score_label = "Model probability (SVM, Platt-scaled — not a safety guarantee)"
    elif decision_score is not None:
        score_type = "decision_score"
        score_value = decision_score
        score_label = "SVM decision score (signed distance to the class boundary)"
    else:
        score_type = "class_only"
        score_value = None
        score_label = "Class prediction only (no probability or decision score available)"

    return {
        "class_idx": predicted_class_idx,
        "proba": proba,
        "decision_score": decision_score,
        "score_type": score_type,
        "score_value": score_value,
        "score_label": score_label,
    }


def predict_with_confidence(model, input_scaled: np.ndarray):
    """
    Make a class prediction and return probabilities when the model provides them.

    Returns:
        Tuple of (predicted_class_idx, probability_array_or_None)
        probability_array is None when predict_proba is not available.
    """
    try:
        result = predict_with_scores(model, input_scaled)
        return result["class_idx"], result["proba"]
    except Exception as e:
        st.error(f"Error computing prediction: {str(e)}")
        return None, None


def calculate_confidence_and_risk(
    confidence: float, predicted_class: str
) -> Tuple[str, str, str]:
    """
    Calculate the risk level based on confidence and predicted class.

    Args:
        confidence: Confidence percentage (0-100)
        predicted_class: "EDIBLE" or "POISONOUS"

    Returns:
        Tuple of (risk_level, risk_color, risk_icon)
    """
    if predicted_class.upper() == "POISONOUS":
        if confidence >= 90:
            risk_level = "Model: poisonous"
            risk_color = "red"
            risk_icon = "🔴"
        elif confidence >= 70:
            risk_level = "Model leans poisonous"
            risk_color = "orange"
            risk_icon = "🟠"
        else:
            risk_level = "Uncertain (do not eat)"
            risk_color = "gray"
            risk_icon = "⚪"
    else:  # EDIBLE
        if confidence >= 90:
            risk_level = "Model: edible"
            risk_color = "green"
            risk_icon = "🟢"
        elif confidence >= 70:
            risk_level = "Model leans edible"
            risk_color = "lightgreen"
            risk_icon = "🟡"
        else:
            risk_level = "Uncertain (do not eat)"
            risk_color = "gray"
            risk_icon = "⚪"

    return risk_level, risk_color, risk_icon


def get_feature_explanations(input_dict: Dict, top_features_list: List) -> None:
    """
    Provide beginner-friendly explanations for top contributing features.

    Args:
        input_dict: Dictionary of feature name -> selected value
        top_features_list: List of top contributing feature names
    """
    explanations = {
        "odor": (
            "The odor of a mushroom is a strong indicator. Certain smells are "
            "commonly associated with poisonous species."
        ),
        "gill-color": (
            "The color of the gills can vary widely and is often a key characteristic "
            "for identification. Some colors are more prevalent in toxic mushrooms."
        ),
        "veil-color": (
            "The color of the veil (a membrane covering the gills when young) can be "
            "a distinguishing feature, sometimes indicating toxicity."
        ),
        "bruises": (
            "Whether a mushroom bruises (changes color when touched) and the color "
            "of the bruise can be an important clue to its edibility."
        ),
        "ring-type": (
            "The type of ring on the stalk (if present) is a morphological feature "
            "that helps differentiate species, some of which are poisonous."
        ),
        "cap-shape": (
            "The shape of the cap (e.g., convex, flat, bell-shaped) is a primary "
            "visual characteristic used in mushroom identification."
        ),
        "cap-surface": (
            "The texture of the cap surface (e.g., fibrous, scaly, smooth) can "
            "provide clues about the mushroom's species."
        ),
        "cap-color": (
            "The color of the cap varies across species and can be a useful "
            "identification feature when combined with other characteristics."
        ),
        "gill-attachment": (
            "How the gills are attached to the stalk is a critical identification feature."
        ),
        "gill-spacing": (
            "The density of the gills (how close they are to each other) is another "
            "morphological detail."
        ),
        "gill-size": (
            "The width of the gills can help distinguish between similar-looking species."
        ),
        "stalk-shape": (
            "The shape of the stalk (e.g., tapering, enlarging) can be indicative of "
            "the species."
        ),
        "stalk-root": (
            "The base of the stalk, or its 'root', can have distinct features that "
            "aid in identification."
        ),
        "stalk-surface-above-ring": (
            "The texture of the stalk above the ring provides additional morphological clues."
        ),
        "stalk-surface-below-ring": (
            "The texture of the stalk below the ring is another distinguishing feature."
        ),
        "stalk-color-above-ring": (
            "The color of the stalk above the ring can vary by species and age."
        ),
        "stalk-color-below-ring": (
            "The color of the stalk below the ring is another identification feature."
        ),
        "veil-type": (
            "The type of veil present on the mushroom is a taxonomic characteristic."
        ),
        "ring-number": (
            "The number of rings on the stalk is a morphological feature that "
            "helps identify the species."
        ),
        "spore-print-color": (
            "The color of the spore print is one of the most reliable features for "
            "mushroom identification, as it is consistent within species."
        ),
        "population": (
            "The way mushrooms grow (e.g., scattered, clustered, solitary) can "
            "sometimes correlate with species and edibility."
        ),
        "habitat": (
            "The environment where a mushroom grows (e.g., woods, grasses, urban) "
            "is a significant factor in identification."
        ),
    }

    for feature_name in top_features_list:
        st.markdown(f"**{feature_name.replace('-', ' ').title()}**")
        st.write(f"Selected value: **{input_dict.get(feature_name, 'N/A')}**")
        st.write(
            explanations.get(
                feature_name,
                "This feature is important for classification, contributing to the model's decision.",
            )
        )
