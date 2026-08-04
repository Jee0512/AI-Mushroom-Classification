"""
Data transformation utilities for the Mushroom Classifier.
Transforms categorical user inputs into scaled numerical arrays for the SVM model.
"""
import numpy as np


def transform_input(input_dict: dict, label_encoders: dict, scaler) -> np.ndarray:
    """
    Transform the user's input dictionary into a scaled numerical array
    suitable for model prediction.

    Args:
        input_dict: Dictionary of feature name -> categorical value
        label_encoders: Dictionary of feature name -> LabelEncoder
        scaler: Fitted StandardScaler

    Returns:
        numpy.ndarray: Scaled input array of shape (1, n_features), or None on error
    """
    try:
        transformed_values = []

        for feature, value in input_dict.items():
            if feature in label_encoders:
                le = label_encoders[feature]

                # Handle missing/unknown values gracefully
                if value not in le.classes_:
                    raise ValueError(f"Unknown value '{value}' for feature '{feature}'")

                transformed_value = le.transform([value])[0]
                transformed_values.append(transformed_value)
            else:
                raise ValueError(f"No encoder found for feature '{feature}'")

        input_array = np.array(transformed_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        return input_scaled

    except Exception as e:
        # Re-raise so the caller can display context-specific errors
        raise ValueError(f"Error in transformation: {str(e)}")


def transform_batch(df, label_encoders, scaler) -> np.ndarray:
    """
    Transform a DataFrame of categorical features into scaled numerical data.
    Useful for SHAP background data and batch predictions.

    Args:
        df: pandas DataFrame with categorical columns
        label_encoders: Dictionary of feature name -> LabelEncoder
        scaler: Fitted StandardScaler

    Returns:
        numpy.ndarray: Scaled array of shape (n_samples, n_features)
    """
    transformed_values = []

    for feature in label_encoders.keys():
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")

        le = label_encoders[feature]

        # Map known values, default to first class for unseen/missing values
        def encode(val):
            if val in le.classes_:
                return le.transform([val])[0]
            return 0  # Fallback to first class

        transformed_values.append(df[feature].map(encode).values)

    # Stack columns into a 2D array
    input_array = np.column_stack(transformed_values)
    return scaler.transform(input_array)
