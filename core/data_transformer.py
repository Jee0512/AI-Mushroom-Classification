"""
Data transformation utilities for the Mushroom Classifier.
Transforms categorical user inputs into scaled numerical arrays for the SVM model.

Feature order is NEVER taken from dict insertion order. Callers must pass
feature_order (from models/feature_order.json or config.FEATURE_ORDER).
UI English labels are mapped back to the original UCI letter codes before
LabelEncoder.transform is applied.
"""
import numpy as np
from config import FEATURE_ORDER, FEATURE_VALUE_MAP


def _normalize_code(value) -> str:
    return str(value).strip()


def ui_value_to_code(feature: str, value) -> str:
    """
    Convert a UI selection (English label or raw UCI code) to the dataset code.
    """
    raw = _normalize_code(value)
    mapping = FEATURE_VALUE_MAP.get(feature, {})
    if raw in mapping:
        return raw
    reverse = {label: code for code, label in mapping.items()}
    if raw in reverse:
        return reverse[raw]
    # Case-insensitive label match
    lower_reverse = {label.lower(): code for code, label in mapping.items()}
    if raw.lower() in lower_reverse:
        return lower_reverse[raw.lower()]
    raise ValueError(f"Unknown value '{value}' for feature '{feature}'")


def code_to_ui_label(feature: str, code) -> str:
    """Convert a UCI dataset code to the English UI label."""
    mapping = FEATURE_VALUE_MAP.get(feature, {})
    key = _normalize_code(code)
    return mapping.get(key, key)


def ui_options_for_encoder(feature: str, encoder) -> list:
    """English labels in the same order as encoder.classes_ (trained codes)."""
    options = []
    for code in encoder.classes_:
        options.append(code_to_ui_label(feature, code))
    return options


def resolve_feature_order(label_encoders: dict, feature_order=None) -> list:
    """Return the single authoritative feature list used by the scaler/SVM."""
    if feature_order:
        order = list(feature_order)
    elif label_encoders:
        # Prefer canonical CSV order when encoders contain those keys.
        order = [f for f in FEATURE_ORDER if f in label_encoders]
        extras = [f for f in label_encoders.keys() if f not in order]
        order.extend(extras)
    else:
        order = list(FEATURE_ORDER)
    missing = [f for f in order if f not in label_encoders]
    if missing:
        raise ValueError(f"Encoders missing features: {missing}")
    return order


def transform_input(input_dict: dict, label_encoders: dict, scaler, feature_order=None) -> np.ndarray:
    """
    Transform the user's input dictionary into a scaled numerical array
    suitable for model prediction.

    Args:
        input_dict: Dictionary of feature name -> UI label or UCI code
        label_encoders: Dictionary of feature name -> LabelEncoder (fit on codes)
        scaler: Fitted StandardScaler
        feature_order: Explicit column order matching training. Required for
            correctness; falls back to FEATURE_ORDER ∩ encoder keys.

    Returns:
        numpy.ndarray: Scaled input array of shape (1, n_features)
    """
    try:
        order = resolve_feature_order(label_encoders, feature_order)
        transformed_values = []

        for feature in order:
            if feature not in input_dict:
                raise ValueError(f"Missing required feature '{feature}'")
            code = ui_value_to_code(feature, input_dict[feature])
            le = label_encoders[feature]
            classes = [_normalize_code(c) for c in le.classes_]
            if code not in classes:
                raise ValueError(
                    f"Unknown encoded value '{code}' for feature '{feature}'. "
                    f"Valid codes: {classes}"
                )
            # Map back to the encoder's native class dtype
            native = le.classes_[classes.index(code)]
            transformed_values.append(le.transform([native])[0])

        input_array = np.array(transformed_values, dtype=float).reshape(1, -1)
        return scaler.transform(input_array)

    except Exception as e:
        raise ValueError(f"Error in transformation: {str(e)}") from e


def transform_batch(df, label_encoders, scaler, feature_order=None) -> np.ndarray:
    """
    Transform a DataFrame of UCI letter-code columns into scaled numerical data.
    """
    order = resolve_feature_order(label_encoders, feature_order)
    transformed_values = []

    for feature in order:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")

        le = label_encoders[feature]
        classes = set(_normalize_code(c) for c in le.classes_)

        def encode(val):
            code = _normalize_code(val)
            if code not in classes:
                raise ValueError(f"Unknown value '{val}' for feature '{feature}'")
            native = le.classes_[[_normalize_code(c) for c in le.classes_].index(code)]
            return le.transform([native])[0]

        transformed_values.append(df[feature].map(encode).values)

    input_array = np.column_stack(transformed_values)
    return scaler.transform(input_array)
