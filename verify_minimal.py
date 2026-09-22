"""Minimal verification that writes results to a file."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

results = []
errors = []

def check(desc, fn):
    try:
        fn()
        results.append(f"✅ {desc}")
    except Exception as e:
        errors.append(f"❌ {desc}: {e}")
        results.append(f"❌ {desc}: {e}")
        traceback.print_exc()

def t_imports():
    import config
    import core.model_loader
    import core.data_transformer
    import core.prediction_utils
    import components.cards
    import components.charts
    import components.theme

    import utils.history_manager
    import utils.export_utils
    import reports.pdf_generator
    import vision.image_processor
    import vision.image_classifier
    import explainability.shap_explainer
    import explainability.grad_cam_explainer

def t_models():
    import pickle
    with open("models/svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    assert model is not None
    assert len(label_encoders) > 0
    assert scaler is not None
    # Test a prediction
    from core.data_transformer import transform_input
    from config import FEATURE_MAPPING
    test_input = {}
    for feature in label_encoders.keys():
        test_input[feature] = FEATURE_MAPPING.get(feature, label_encoders[feature].classes_)[0]
    
    result = transform_input(test_input, label_encoders, scaler)
    from core.prediction_utils import predict_with_confidence
    pred_idx, proba = predict_with_confidence(model, result)
    assert proba is not None

def t_pdf():
    from reports.pdf_generator import generate_safety_report
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (50, 50), color=(46, 125, 50))
    data = {
        "Prediction": "EDIBLE",
        "Confidence": "95.5%",
        "Risk Level": "Safe",
        "Model Used": "SVM",
        "Prediction Time": "0.1s",
        "Disclaimer": "Test",
        "Top Contributing Features": ["odor", "gill-color"],
        "Image": img,
    }
    pdf = generate_safety_report(data)
    assert len(pdf) > 0

def t_export():
    from utils.export_utils import export_to_csv, export_to_json
    data = [{"timestamp": "2024-01-01", "prediction": "EDIBLE"}]
    assert len(export_to_csv(data)) > 0
    assert len(export_to_json(data)) > 0

def t_charts():
    import matplotlib
    matplotlib.use("Agg")
    from components.charts import confidence_gauge, probability_bar_chart
    import numpy as np
    fig1 = confidence_gauge(85.5, "EDIBLE")
    fig2 = probability_bar_chart(np.array([0.855, 0.145]))

def t_history():
    from utils.history_manager import init_db, save_prediction, get_all_history
    init_db()
    save_prediction({
        "timestamp": "2024-01-01 12:00:00",
        "method": "Manual",
        "prediction": "EDIBLE",
        "confidence": "95%",
        "risk_level": "Safe",
        "features": {"odor": "almond"},
        "model": "SVM",
    })
    df = get_all_history()
    assert len(df) > 0

# Run checks
check("Module imports", t_imports)
check("Model loading + prediction", t_models)
check("PDF generation", t_pdf)
check("Export utilities", t_export)
check("Chart generation", t_charts)
check("History manager", t_history)

# Write results
out = ROOT / "verification_result.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("🍄 Mushroom Classifier - Verification Results\n")
    f.write("=" * 60 + "\n\n")
    for r in results:
        f.write(r + "\n")
    f.write("\n" + "=" * 60 + "\n")
    if errors:
        f.write(f"❌ {len(errors)} ERROR(S) FOUND\n")
    else:
        f.write("✅ ALL CHECKS PASSED\n")
    f.write("=" * 60 + "\n")

print("Verification complete. Results written to verification_result.txt")

