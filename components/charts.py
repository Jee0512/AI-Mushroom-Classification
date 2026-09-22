"""
Chart components for the Mushroom Classifier.
Provides reusable chart functions for visualizations.
"""
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from config import SUCCESS_COLOR, DANGER_COLOR, WARNING_COLOR, INFO_COLOR, PRIMARY_COLOR


def configure_matplotlib():
    """Configure matplotlib for consistent styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def confidence_gauge(confidence: float, predicted_class: str, figsize=(4, 3)):
    """
    Draw a circular gauge chart showing confidence level.

    Args:
        confidence: Confidence percentage (0-100)
        predicted_class: "EDIBLE" or "POISONOUS"
        figsize: Figure size tuple
    Returns:
        matplotlib figure
    """
    configure_matplotlib()

    is_edible = predicted_class.upper() == "EDIBLE"
    color = SUCCESS_COLOR if is_edible else DANGER_COLOR

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)

    # Set angle range: start at 180° (left), go clockwise 180°
    theta_start = np.pi
    theta_end = 2 * np.pi

    # Background arc (full 180°)
    theta_bg = np.linspace(theta_start, theta_end, 100)
    ax.fill_between(theta_bg, 0, 1, color="#E0E0E0", alpha=0.3)

    # Confidence arc
    confidence_rad = confidence / 100 * np.pi
    theta_conf = np.linspace(theta_start, theta_start + confidence_rad, 50)
    ax.fill_between(theta_conf, 0, 1, color=color, alpha=0.8)

    # Add a border to the filled arc
    ax.plot(theta_conf, [1] * len(theta_conf), color=color, linewidth=2)

    # Remove labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.5)
    ax.spines["polar"].set_visible(False)

    # Add center text
    ax.text(
        0, 0, f"{confidence:.1f}%",
        ha="center", va="center",
        fontsize=24, fontweight="bold",
        color=color,
        transform=ax.transData
    )

    ax.text(
        0, -0.3, "Confidence",
        ha="center", va="center",
        fontsize=10, color="#666666",
        transform=ax.transData
    )

    return fig


def probability_bar_chart(probabilities: np.ndarray, figsize=(6, 2)):
    """
    Draw a horizontal bar chart showing class probabilities.

    Args:
        probabilities: Array of [edible_prob, poisonous_prob]
        figsize: Figure size tuple
    Returns:
        matplotlib figure
    """
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)

    classes = ["Edible 🍄", "Poisonous ☠️"]
    colors = [SUCCESS_COLOR, DANGER_COLOR]

    y_pos = [0, 1]
    bars = ax.barh(y_pos, probabilities, color=colors, height=0.5, edgecolor="white", linewidth=1.5)

    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        if prob > 0.05:
            ax.text(
                prob + 0.01, i,
                f"{prob*100:.1f}%",
                va="center", fontsize=11, fontweight="bold",
                color=colors[i]
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Probability", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Class Probabilities", fontsize=12, fontweight="bold", pad=10)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def feature_importance_chart(feature_names, importance_values, top_n=10, figsize=(8, 5)):
    """
    Draw a horizontal bar chart of feature importance.

    Args:
        feature_names: List of feature names
        importance_values: List of importance values (SHAP values)
        top_n: Number of top features to show
        figsize: Figure size tuple
    Returns:
        matplotlib figure
    """
    configure_matplotlib()

    # Sort by absolute importance
    sorted_idx = np.argsort(np.abs(importance_values))[::-1]
    top_idx = sorted_idx[:top_n]

    names = [feature_names[i].replace("-", " ").title() for i in top_idx][::-1]
    values = importance_values[top_idx][::-1]
    colors = [SUCCESS_COLOR if v > 0 else DANGER_COLOR for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=1.2)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=10)
    ax.set_title("Top Feature Contributions", fontsize=12, fontweight="bold", pad=10)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = val + 0.001 if val >= 0 else val - 0.01
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, i, f"{val:.4f}", va="center", fontsize=8, ha=ha)

    # Legend
    edible_patch = mpatches.Patch(color=SUCCESS_COLOR, label="Supports Edible")
    poisonous_patch = mpatches.Patch(color=DANGER_COLOR, label="Supports Poisonous")
    ax.legend(handles=[edible_patch, poisonous_patch], loc="lower right", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def prediction_trend_chart(history_df: pd.DataFrame, figsize=(10, 4)):
    """
    Draw a line chart showing prediction trends over time.

    Args:
        history_df: DataFrame with 'timestamp' and 'prediction' columns
        figsize: Figure size tuple
    Returns:
        matplotlib figure
    """
    configure_matplotlib()

    if history_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No prediction data available", ha="center", va="center", fontsize=12)
        return fig

    # Ensure timestamp is datetime
    history_df = history_df.copy()
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    history_df["date"] = history_df["timestamp"].dt.date

    # Count predictions per day
    daily_counts = history_df.groupby(["date", "prediction"]).size().unstack(fill_value=0)

    # History stores predictions as "EDIBLE"/"POISONOUS" (uppercase).
    # Normalize column names so the trend lines render correctly.
    daily_counts.columns = [str(c).upper() for c in daily_counts.columns]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)

    if "EDIBLE" in daily_counts.columns:
        ax.plot(daily_counts.index, daily_counts["EDIBLE"], color=SUCCESS_COLOR,
                marker="o", linewidth=2, label="Edible")
    if "POISONOUS" in daily_counts.columns:
        ax.plot(daily_counts.index, daily_counts["POISONOUS"], color=DANGER_COLOR,
                marker="o", linewidth=2, label="Poisonous")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Number of Predictions", fontsize=10)
    ax.set_title("Prediction Trends Over Time", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def confusion_matrix_plot(cm, classes, figsize=(6, 5)):
    """
    Draw a confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2x2 array)
        classes: List of class names
        figsize: Figure size tuple
    Returns:
        matplotlib figure
    """
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)

    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_norm, cmap="Greens", interpolation="nearest")

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm_norm[i, j] > 0.5 else "black")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def roc_curve_plot(fpr, tpr, auc_score, figsize=(6, 5)):
    """
    Draw an ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        figsize: Figure size tuple
    Returns:
        matplotlib figure
    """
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)

    ax.plot(fpr, tpr, color=PRIMARY_COLOR, linewidth=2.5, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color=PRIMARY_COLOR)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def pie_chart(labels, values, colors, title="Distribution", figsize=(5, 4)):
    """
    Draw a pie chart.

    Args:
        labels: List of labels
        values: List of values
        colors: List of colors
        title: Chart title
        figsize: Figure size tuple
    Returns:
        matplotlib figure
    """
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    return fig
