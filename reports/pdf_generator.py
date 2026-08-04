"""
PDF report generation for the Mushroom Classifier.
Creates professional safety reports with prediction results, explanations,
charts, and metadata using ReportLab.
"""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor, white
import io
from datetime import datetime
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Color Palette
PRIMARY = HexColor("#2E7D32")
PRIMARY_LIGHT = HexColor("#4CAF50")
DANGER = HexColor("#D32F2F")
WARNING = HexColor("#F57C00")
TEXT_DARK = HexColor("#212121")
TEXT_MED = HexColor("#555555")
TEXT_LIGHT = HexColor("#888888")


def _create_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    # Helper to add a style only if it doesn't already exist
    def add_style(name, **kwargs):
        if name not in styles:
            styles.add(ParagraphStyle(name=name, **kwargs))
        return styles[name]

    add_style(
        "ReportTitle",
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        textColor=PRIMARY,
    )
        "ReportSubtitle",
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        fontName="Helvetica",
        textColor=TEXT_MED,
        spaceAfter=12,
    )
    add_style(
        "SectionHeading",
        fontSize=16,
        leading=20,
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=8,
        textColor=PRIMARY,
    )
    add_style(
        "SubHeading",
        fontSize=13,
        leading=16,
        fontName="Helvetica-Bold",
        spaceAfter=6,
        textColor=TEXT_DARK,
    )
    # Override the built-in BodyText style rather than re-adding it
    styles["BodyText"] = ParagraphStyle(
        name="BodyText",
        fontSize=10,
        leading=13,
        fontName="Helvetica",
        spaceAfter=6,
        textColor=TEXT_MED,
    )
    add_style(
        "DisclaimerText",
        fontSize=8,
        leading=11,
        fontName="Helvetica-Oblique",
        textColor=TEXT_LIGHT,
        spaceBefore=16,
        spaceAfter=8,
    )

    return styles
def _image_to_buffer(img: PILImage.Image, max_size=(600, 600)) -> io.BytesIO:
    """Convert a PIL image to a BytesIO buffer, resized to fit."""
    img = img.copy()
    img.thumbnail(max_size, PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _create_confidence_chart(confidence: float, predicted_class: str) -> io.BytesIO:
    """Create a confidence bar chart as an image buffer."""
    is_edible = predicted_class.upper() == "EDIBLE"
    color = PRIMARY_LIGHT if is_edible else DANGER

    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.patch.set_alpha(0)

    edible_prob = confidence / 100 if is_edible else 1 - confidence / 100
    poisonous_prob = 1 - edible_prob

    ax.barh(["Edible", "Poisonous"], [edible_prob, poisonous_prob],
            color=[PRIMARY_LIGHT, DANGER], height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.grid(axis="x", alpha=0.3)

    for i, (prob, label) in enumerate(zip([edible_prob, poisonous_prob], ["Edible", "Poisonous"])):
        ax.text(prob + 0.01, i, f"{prob*100:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _create_shap_chart(feature_names, shap_values) -> io.BytesIO:
    """Create a SHAP feature importance chart as an image buffer."""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_alpha(0)

    impacts = np.abs(shap_values)
    sorted_idx = np.argsort(impacts)[::-1][:8]
    names = [feature_names[i].replace("-", " ").title() for i in sorted_idx][::-1]
    values = shap_values[sorted_idx][::-1]

    colors = [PRIMARY_LIGHT if v > 0 else DANGER for v in values]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("SHAP Value", fontsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_safety_report(report_data: dict) -> bytes:
    """
    Generate a professional PDF safety report.

    Args:
        report_data: Dictionary containing:
            - Prediction, Confidence, Risk Level, Model Used, Prediction Time
            - Input Features (optional dict)
            - Image (optional PIL image)
            - Top Contributing Features (optional list)
            - Top Visual Features (optional list)
            - Shap Values (optional array), Feature Names (optional list)
            - Disclaimer

    Returns:
        bytes: PDF content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = _create_styles()
    story = []

    # ── Title ─────────────────────────────────────────────────────
    story.append(Paragraph("🍄 Mushroom Edibility Safety Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["ReportSubtitle"],
    ))
    story.append(Spacer(1, 0.2 * inch))

    # ── Prediction Summary ─────────────────────────────────────────
    story.append(Paragraph("Prediction Summary", styles["SectionHeading"]))

    summary_data = [
        ["Prediction", report_data.get("Prediction", "N/A")],
        ["Confidence", report_data.get("Confidence", "N/A")],
        ["Risk Level", report_data.get("Risk Level", "N/A")],
        ["Model Used", report_data.get("Model Used", "N/A")],
        ["Prediction Time", report_data.get("Prediction Time", "N/A")],
    ]

    summary_table = Table(summary_data, colWidths=[1.5 * inch, 4.5 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PRIMARY),
        ("TEXTCOLOR", (0, 0), (0, -1), white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F5F5F5")]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    # ── Confidence Chart ───────────────────────────────────────────
    confidence_str = report_data.get("Confidence", "0%").replace("%", "")
    try:
        confidence = float(confidence_str)
        pred_class = report_data.get("Prediction", "UNKNOWN")
        chart_buf = _create_confidence_chart(confidence, pred_class)
        chart_img = RLImage(chart_buf)
        chart_img.drawWidth = 5.5 * inch
        chart_img.drawHeight = 1.5 * inch
        story.append(Paragraph("Confidence Distribution", styles["SubHeading"]))
        story.append(chart_img)
        story.append(Spacer(1, 0.2 * inch))
    except Exception:
        pass

    # ── Input Features ─────────────────────────────────────────────
    if report_data.get("Input Features"):
        story.append(Paragraph("Input Features (Manual Prediction)", styles["SectionHeading"]))
        features = report_data["Input Features"]
        feature_data = [["Feature", "Value"]]
        for feat, val in features.items():
            feature_data.append([feat.replace("-", " ").title(), str(val)])

        feature_table = Table(feature_data, colWidths=[3 * inch, 3 * inch])
        feature_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_LIGHT),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F5F5F5")]),
        ]))
        story.append(feature_table)
        story.append(Spacer(1, 0.2 * inch))

    # ── Uploaded Image ─────────────────────────────────────────────
    if report_data.get("Image") and isinstance(report_data["Image"], PILImage.Image):
        story.append(Paragraph("Uploaded Mushroom Image", styles["SectionHeading"]))
        img_buf = _image_to_buffer(report_data["Image"], max_size=(300, 300))
        img = RLImage(img_buf)
        img.drawWidth = 2.5 * inch
        img.drawHeight = 2.5 * inch
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))

    # ── Explainability Insights ────────────────────────────────────
    story.append(Paragraph("Explainability Insights", styles["SectionHeading"]))

    # SHAP chart if available
    shap_values = report_data.get("Shap Values")
    feature_names = report_data.get("Feature Names")
    if shap_values is not None and feature_names is not None:
        try:
            shap_buf = _create_shap_chart(feature_names, shap_values)
            shap_img = RLImage(shap_buf)
            shap_img.drawWidth = 5.5 * inch
            shap_img.drawHeight = 3 * inch
            story.append(Paragraph("Feature Importance (SHAP)", styles["SubHeading"]))
            story.append(shap_img)
            story.append(Spacer(1, 0.1 * inch))
        except Exception:
            pass

    if report_data.get("Top Contributing Features"):
        story.append(Paragraph("Top Contributing Features:", styles["SubHeading"]))
        for feature in report_data["Top Contributing Features"]:
            story.append(Paragraph(f"• {feature}", styles["BodyText"]))

    if report_data.get("Top Visual Features"):
        story.append(Paragraph("Top Visual Features Detected (Image Prediction):", styles["SubHeading"]))
        for feature in report_data["Top Visual Features"]:
            story.append(Paragraph(f"• {feature}", styles["BodyText"]))

    story.append(Spacer(1, 0.2 * inch))

    # ── Disclaimer ─────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Disclaimer", styles["SectionHeading"]))
    story.append(Paragraph(
        report_data.get(
            "Disclaimer",
            "This report is generated by an AI model and should not be used as the sole "
            "basis for determining if a mushroom is safe to eat. Always consult with "
            "mycology experts.",
        ),
        styles["DisclaimerText"],
    ))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Model Version: 1.0", styles["BodyText"]))
    story.append(Paragraph(
        f"Report generated by Mushroom Classifier v2.0.0",
        styles["DisclaimerText"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
