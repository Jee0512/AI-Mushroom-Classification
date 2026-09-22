"""
Central configuration for the Mushroom Classifier application.
All paths, model settings, and app constants are defined here.
"""
import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ── Data Paths ─────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
REPORTS_DIR = ROOT_DIR / "reports"
DEPLOYMENT_DIR = ROOT_DIR / "deployment"

# ── Model Files ────────────────────────────────────────────────────
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_ORDER_PATH = MODELS_DIR / "feature_order.json"
TRAINING_METRICS_PATH = MODELS_DIR / "training_metrics.json"
IMAGE_MODEL_PATH = MODELS_DIR / "efficientnet_model.h5"
PYTORCH_IMAGE_MODEL_PATH = MODELS_DIR / "image_model.pt"
SKLEARN_IMAGE_MODEL_PATH = MODELS_DIR / "image_model.pkl"

# ── Authoritative feature order (UCI mushrooms.csv column order, excluding class)
FEATURE_ORDER = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

FEATURE_GROUPS = {
    "Cap": ["cap-shape", "cap-surface", "cap-color"],
    "Gills": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
    "Stalk": [
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
    ],
    "Veil & Ring": ["veil-type", "veil-color", "ring-number", "ring-type"],
    "Other": ["bruises", "odor", "spore-print-color", "population", "habitat"],
}

SAFETY_DISCLAIMER = (
    "This tool is for educational and research purposes only. "
    "Do NOT use it as the sole basis for deciding whether a wild mushroom is safe to eat. "
    "Many poisonous species resemble edible ones. Always consult a qualified mycologist."
)

# ── Dataset ────────────────────────────────────────────────────────
DATASET_PATH = DATA_DIR / "mushrooms.csv"
DATASET_URL = "https://www.kaggle.com/datasets/uciml/mushroom-classification"

# ── App Metadata ───────────────────────────────────────────────────
APP_NAME = "🍄 Mushroom Classifier"
APP_TAGLINE = "AI-Powered Mushroom Edibility Analysis"
APP_VERSION = "2.0.0"
APP_AUTHOR = "ML Engineering Team"
APP_DESCRIPTION = (
    "A production-grade machine learning application that classifies mushrooms "
    "as edible or poisonous using a Support Vector Machine (SVM) model and "
    "deep learning-based image recognition."
)

# ── Streamlit Settings ─────────────────────────────────────────────
PAGE_TITLE = "Mushroom Classifier | AI Edibility Analysis"
PAGE_ICON = "🍄"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# ── Theme Colors ───────────────────────────────────────────────────
PRIMARY_COLOR = "#4CAF50"
SECONDARY_COLOR = "#2196F3"
ACCENT_COLOR = "#FF9800"
DANGER_COLOR = "#f44336"
SUCCESS_COLOR = "#4CAF50"
WARNING_COLOR = "#FF9800"
INFO_COLOR = "#2196F3"
DARK_BG = "#1E1E1E"
DARK_CARD = "#2D2D2D"
LIGHT_BG = "#F5F5F5"
LIGHT_CARD = "#FFFFFF"

# ── Feature Names (for display) ────────────────────────────────────
FEATURE_DISPLAY_NAMES = {
    "cap-shape": "Cap Shape",
    "cap-surface": "Cap Surface",
    "cap-color": "Cap Color",
    "bruises": "Bruises",
    "odor": "Odor",
    "gill-attachment": "Gill Attachment",
    "gill-spacing": "Gill Spacing",
    "gill-size": "Gill Size",
    "gill-color": "Gill Color",
    "stalk-shape": "Stalk Shape",
    "stalk-root": "Stalk Root",
    "stalk-surface-above-ring": "Stalk Surface (Above Ring)",
    "stalk-surface-below-ring": "Stalk Surface (Below Ring)",
    "stalk-color-above-ring": "Stalk Color (Above Ring)",
    "stalk-color-below-ring": "Stalk Color (Below Ring)",
    "veil-type": "Veil Type",
    "veil-color": "Veil Color",
    "ring-number": "Ring Number",
    "ring-type": "Ring Type",
    "spore-print-color": "Spore Print Color",
    "population": "Population",
    "habitat": "Habitat",
}

# ── Dataset codes (UCI) → human-readable UI labels ─────────────────
# Keys are the actual CSV values. Training fits LabelEncoders on these codes.
FEATURE_VALUE_MAP = {
    "cap-shape": {
        "b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat", "k": "Knobbed", "s": "Sunken",
    },
    "cap-surface": {"f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"},
    "cap-color": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "r": "Green",
        "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow",
    },
    "bruises": {"t": "Yes", "f": "No"},
    "odor": {
        "a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul",
        "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy",
    },
    "gill-attachment": {"a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"},
    "gill-spacing": {"c": "Close", "w": "Crowded", "d": "Distant"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "gill-color": {
        "k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Gray", "r": "Green",
        "o": "Orange", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow",
    },
    "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk-root": {
        "b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal",
        "z": "Rhizomorphs", "r": "Rooted", "?": "Missing",
    },
    "stalk-surface-above-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk-surface-below-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk-color-above-ring": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange",
        "p": "Pink", "e": "Red", "w": "White", "y": "Yellow",
    },
    "stalk-color-below-ring": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange",
        "p": "Pink", "e": "Red", "w": "White", "y": "Yellow",
    },
    "veil-type": {"p": "Partial", "u": "Universal"},
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"n": "None", "o": "One", "t": "Two"},
    "ring-type": {
        "c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large",
        "n": "None", "p": "Pendant", "s": "Sheathing", "z": "Zone",
    },
    "spore-print-color": {
        "k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green",
        "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow",
    },
    "population": {
        "a": "Abundant", "c": "Clustered", "n": "Numerous",
        "s": "Scattered", "v": "Several", "y": "Solitary",
    },
    "habitat": {
        "g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths",
        "u": "Urban", "w": "Waste", "d": "Woods",
    },
}

# English labels for each feature (dict insertion order). Prefer encoder.classes_
# mapped through FEATURE_VALUE_MAP when building UI options.
FEATURE_MAPPING = {
    feature: list(mapping.values())
    for feature, mapping in FEATURE_VALUE_MAP.items()
}

# ── Feature Descriptions (for encyclopedia) ────────────────────────
FEATURE_DESCRIPTIONS = {
    "cap-shape": "The shape of the mushroom cap (e.g., bell, conical, convex, flat, knobbed, sunken). This is a primary visual characteristic used in mushroom identification.",
    "cap-surface": "The texture of the cap's surface (e.g., fibrous, grooved, scaly, smooth). Different species have distinct surface textures.",
    "cap-color": "The color of the cap (e.g., brown, buff, cinnamon, gray, green, pink, purple, red, white, yellow). Color can vary with age and environment.",
    "bruises": "Whether the mushroom flesh changes color when bruised or cut. This is a key identification feature for many species.",
    "odor": "The smell of the mushroom (e.g., almond, anise, creosote, fishy, foul, musty, none, pungent, spicy). Odor is one of the most reliable indicators of edibility.",
    "gill-attachment": "How the gills are attached to the stalk (e.g., attached, descending, free, notched). This is a critical taxonomic feature.",
    "gill-spacing": "The distance between gills (close or crowded). Gill density helps distinguish between similar species.",
    "gill-size": "The width of the gills (broad or narrow). Gill size can vary within a species.",
    "gill-color": "The color of the gills (e.g., black, brown, buff, chocolate, gray, green, orange, pink, purple, red, white, yellow). Gill color changes with spore maturity.",
    "stalk-shape": "The shape of the stalk (e.g., enlarging, tapering). Stalk morphology aids in identification.",
    "stalk-root": "The base of the stalk (e.g., bulbous, club, cup, equal, rhizomorphs, rooted, missing). Root structure can be diagnostic.",
    "stalk-surface-above-ring": "The texture of the stalk surface above the ring (e.g., fibrous, scaly, silky, smooth).",
    "stalk-surface-below-ring": "The texture of the stalk surface below the ring (e.g., fibrous, scaly, silky, smooth).",
    "stalk-color-above-ring": "The color of the stalk above the ring. Color variations help distinguish species.",
    "stalk-color-below-ring": "The color of the stalk below the ring. Often differs from the upper stalk.",
    "veil-type": "The type of veil (partial or universal). The veil protects the developing gills.",
    "veil-color": "The color of the veil. Veil remnants on the cap or stalk can aid identification.",
    "ring-number": "The number of rings on the stalk (none, one, two). Ring characteristics are important for identification.",
    "ring-type": "The type of ring on the stalk (e.g., cobwebby, evanescent, flaring, large, pendant, sheathing, zone). Ring morphology is a key taxonomic feature.",
    "spore-print-color": "The color of the spore print (e.g., black, brown, buff, chocolate, green, orange, purple, white, yellow). Spore print color is one of the most reliable identification features.",
    "population": "How the mushrooms grow in the wild (e.g., abundant, clustered, numerous, scattered, several, solitary). Growth pattern provides habitat clues.",
    "habitat": "The environment where the mushroom grows (e.g., grasses, leaves, meadows, paths, urban, waste, woods). Habitat is crucial for species identification.",
}

# ── Class Labels ───────────────────────────────────────────────────
CLASS_LABELS = {0: "Edible", 1: "Poisonous"}
CLASS_EMOJIS = {0: "🍄", 1: "☠️"}
CLASS_COLORS = {0: SUCCESS_COLOR, 1: DANGER_COLOR}

# ── Image Settings ─────────────────────────────────────────────────
IMAGE_TARGET_SIZE = (224, 224)
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
MAX_IMAGE_SIZE_MB = 10
ORANGE_HERO_PATH = ASSETS_DIR / "orange_hero.jpg"
GREEN_CENTER_PATH = ASSETS_DIR / "green_center.jpg"

# ── History Settings ───────────────────────────────────────────────
HISTORY_DB_PATH = ROOT_DIR / "data" / "prediction_history.db"
MAX_HISTORY_RECORDS = 1000

# ── Page Configuration ─────────────────────────────────────────────
PAGES = [
    {"name": "Home", "icon": "🏠", "path": "pages/0_home.py", "tooltip": "Go to the home page"},
    {"name": "Select Method", "icon": "⚡", "path": "pages/1_select_method.py", "tooltip": "Choose classification method"},
    {"name": "Feature Classification", "icon": "🔬", "path": "pages/2_manual.py", "tooltip": "Classify a mushroom by selecting its physical features"},
    {"name": "Image Classification", "icon": "📸", "path": "pages/3_image.py", "tooltip": "Upload an image for classification"},
    {"name": "History", "icon": "📜", "path": "pages/4_history.py", "tooltip": "Review past predictions"},
    {"name": "Analytics", "icon": "📊", "path": "pages/5_analytics.py", "tooltip": "View prediction trends"},
    {"name": "About & Settings", "icon": "ℹ️", "path": "pages/6_about.py", "tooltip": "Information about the project and model"},
]

# ── Ensure required directories exist ──────────────────────────────
for dir_path in [DATA_DIR, MODELS_DIR, ASSETS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

