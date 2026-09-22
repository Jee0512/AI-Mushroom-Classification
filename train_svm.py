"""
Train the SVM mushroom classifier from data/mushrooms.csv.

Produces:
  models/svm_model.pkl
  models/label_encoders.pkl
  models/scaler.pkl
  models/feature_order.json
  models/training_metrics.json

Run from the project root:
  python train_svm.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import (
    DATASET_PATH,
    FEATURE_ORDER,
    FEATURE_ORDER_PATH,
    LABEL_ENCODERS_PATH,
    MODELS_DIR,
    SCALER_PATH,
    SVM_MODEL_PATH,
    TRAINING_METRICS_PATH,
)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if "class" not in df.columns:
        raise ValueError("Dataset must contain a 'class' column")

    missing = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected feature columns: {missing}")

    X = df[FEATURE_ORDER].copy()
    y_raw = df["class"].astype(str).str.strip().str.lower()
    # UCI: e = edible, p = poisonous
    y = y_raw.map({"e": 0, "p": 1})
    if y.isna().any():
        unknown = sorted(y_raw[y.isna()].unique().tolist())
        raise ValueError(f"Unexpected class labels: {unknown}")
    y = y.astype(int)

    label_encoders: dict[str, LabelEncoder] = {}
    X_encoded = np.zeros((len(X), len(FEATURE_ORDER)), dtype=float)
    for i, feature in enumerate(FEATURE_ORDER):
        le = LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[feature].astype(str).str.strip())
        label_encoders[feature] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y.to_numpy(), test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    poisonous_idx = list(model.classes_).index(1)

    metrics = {
        "n_samples": int(len(df)),
        "n_features": len(FEATURE_ORDER),
        "feature_order": FEATURE_ORDER,
        "class_mapping": {"e": 0, "p": 1},
        "test_size": 0.2,
        "random_state": 42,
        "model": "SVC(kernel=rbf, C=10, probability=True)",
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, pos_label=1)),
        "test_recall": float(recall_score(y_test, y_pred, pos_label=1)),
        "test_f1": float(f1_score(y_test, y_pred, pos_label=1)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba[:, poisonous_idx])),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(LABEL_ENCODERS_PATH, "wb") as f:
        pickle.dump(label_encoders, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    feature_meta = {
        "feature_order": FEATURE_ORDER,
        "class_mapping": {"e": 0, "p": 1},
        "class_labels": ["EDIBLE", "POISONOUS"],
        "encoder_classes": {
            feature: [str(c) for c in le.classes_]
            for feature, le in label_encoders.items()
        },
    }
    FEATURE_ORDER_PATH.write_text(json.dumps(feature_meta, indent=2), encoding="utf-8")
    TRAINING_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Sanity check: first row round-trip through saved artifacts
    from core.data_transformer import transform_input, code_to_ui_label

    row0 = {feat: str(df.iloc[0][feat]).strip() for feat in FEATURE_ORDER}
    ui_row = {feat: code_to_ui_label(feat, code) for feat, code in row0.items()}
    scaled = transform_input(ui_row, label_encoders, scaler, FEATURE_ORDER)
    pred = int(model.predict(scaled)[0])
    expected = int(y.iloc[0])
    if pred != expected:
        raise RuntimeError(
            f"Sanity check failed on row 0: predicted {pred}, expected {expected}"
        )

    print("Training complete.")
    print(f"  Saved {SVM_MODEL_PATH}")
    print(f"  Saved {LABEL_ENCODERS_PATH}")
    print(f"  Saved {SCALER_PATH}")
    print(f"  Saved {FEATURE_ORDER_PATH}")
    print(f"  Saved {TRAINING_METRICS_PATH}")
    print(
        "  Held-out test: "
        f"acc={metrics['test_accuracy']:.4f}  "
        f"precision={metrics['test_precision']:.4f}  "
        f"recall={metrics['test_recall']:.4f}  "
        f"f1={metrics['test_f1']:.4f}  "
        f"roc_auc={metrics['test_roc_auc']:.4f}"
    )
    print("  Row-0 sanity check passed.")


if __name__ == "__main__":
    main()
