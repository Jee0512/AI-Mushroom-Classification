"""
Reusable card components for the Mushroom Classifier UI.
Provides beautifully styled metric cards, info cards, and feature cards.
"""
import streamlit as st
from config import PRIMARY_COLOR, SUCCESS_COLOR, DANGER_COLOR, WARNING_COLOR, INFO_COLOR


def metric_card(
    label: str,
    value: str,
    delta: str = None,
    icon: str = "📊",
    color: str = PRIMARY_COLOR,
    help_text: str = None,
):
    """
    Display a professional metric card with icon, label, value, and optional delta.
    
    Args:
        label: Metric label
        value: Metric value (string)
        delta: Optional change indicator (e.g., "+5%")
        icon: Emoji icon
        color: Accent color (hex)
        help_text: Optional tooltip text
    """
    help_html = f'<span class="tooltip-icon" title="{help_text}">?</span>' if help_text else ""
    
    st.markdown(
        f"""
        <div class="card card-metric" style="border-top: 3px solid {color};">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            <div class="metric-label">{label}{help_html}</div>
            {f'<div style="font-size: 0.8rem; color: {"#4CAF50" if delta and "+" in delta else "#f44336"}; margin-top: 0.25rem;">{delta}</div>' if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, content: str, icon: str = "ℹ️", color: str = INFO_COLOR):
    """
    Display an informational card.
    
    Args:
        title: Card title
        content: Card body content (markdown supported)
        icon: Emoji icon
        color: Accent color
    """
    st.markdown(
        f"""
        <div class="card" style="border-left: 4px solid {color};">
            <div class="card-header">
                <span>{icon}</span>
                <span>{title}</span>
            </div>
            <div style="color: var(--text-secondary); line-height: 1.6;">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(
    feature_name: str,
    feature_value: str,
    importance: float = None,
    description: str = None,
):
    """
    Display a feature with its value and optional importance.
    
    Args:
        feature_name: Name of the feature
        feature_value: Selected value
        importance: Optional importance score (0-1)
        description: Optional feature description
    """
    display_name = feature_name.replace("-", " ").title()
    
    # Importance bar color
    if importance is not None:
        bar_color = SUCCESS_COLOR if importance > 0.5 else WARNING_COLOR
        bar_width = min(abs(importance) * 100, 100)
    else:
        bar_color = INFO_COLOR
        bar_width = 0
    
    importance_html = ""
    if importance is not None:
        importance_html = f"""
        <div style="margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-light);">
                <span>Importance</span>
                <span>{abs(importance):.3f}</span>
            </div>
            <div style="background: var(--border); border-radius: 10px; height: 6px; overflow: hidden;">
                <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;"></div>
            </div>
        </div>
        """
    
    desc_html = f'<div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.25rem;">{description}</div>' if description else ""
    
    st.markdown(
        f"""
        <div class="card" style="padding: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-weight: 600; font-size: 0.95rem;">{display_name}</div>
                    {desc_html}
                </div>
                <div style="background: {INFO_COLOR}; color: white; border-radius: 20px; padding: 0.2rem 0.75rem; font-size: 0.8rem; font-weight: 600;">
                    {feature_value}
                </div>
            </div>
            {importance_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str, size: str = "sm"):
    """
    Render a status badge.
    
    Args:
        status: "edible", "poisonous", "safe", "risk", etc.
        size: "sm" or "lg"
    """
    status_lower = status.lower()
    
    if status_lower in ["edible", "safe", "low risk"]:
        badge_class = "badge-success"
        icon = "✅"
    elif status_lower in ["poisonous", "high risk", "danger"]:
        badge_class = "badge-danger"
        icon = "⚠️"
    elif status_lower in ["medium risk", "uncertain"]:
        badge_class = "badge-warning"
        icon = "⚡"
    else:
        badge_class = "badge-info"
        icon = "ℹ️"
    
    font_size = "0.9rem" if size == "lg" else "0.8rem"
    padding = "0.35rem 1rem" if size == "lg" else "0.25rem 0.75rem"
    
    st.markdown(
        f"""
        <span class="badge {badge_class}" style="font-size: {font_size}; padding: {padding};">
            {icon} {status}
        </span>
        """,
        unsafe_allow_html=True,
    )


def prediction_result_card(predicted_class: str, confidence: float, risk_level: str):
    """
    Display a comprehensive prediction result card.
    
    Args:
        predicted_class: "EDIBLE" or "POISONOUS"
        confidence: Confidence percentage (0-100)
        risk_level: Risk level string
    """
    is_edible = predicted_class.upper() == "EDIBLE"
    emoji = "🍄" if is_edible else "☠️"
    color = SUCCESS_COLOR if is_edible else DANGER_COLOR
    bg_color = "rgba(46, 125, 50, 0.1)" if is_edible else "rgba(211, 47, 47, 0.1)"
    border_color = SUCCESS_COLOR if is_edible else DANGER_COLOR
    
    st.markdown(
        f"""
        <div class="card" style="text-align: center; border: 2px solid {border_color}; background: {bg_color};">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">{emoji}</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">
                {predicted_class}
            </div>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                <div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: {color};">{confidence:.1f}%</div>
                    <div style="font-size: 0.8rem; color: var(--text-light);">Confidence</div>
                </div>
                <div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: {color};">{risk_level}</div>
                    <div style="font-size: 0.8rem; color: var(--text-light);">Risk Level</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
