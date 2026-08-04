"""
Fast verification - tests core modules without TensorFlow imports.
"""
import sys
import traceback

print("=" * 60)
print("🍄 Mushroom Classifier - Fast Verification (No TF)")
print("=" * 60)

errors = []

def check(module_name, label=None):
    try:
        __import__(module_name)
        print(f"✅ {label or module_name}")
        return True
    except Exception as e:
        errors.append(f"{label or module_name}: {e}")
        print(f"❌ {label or module_name}: {e}")
        return False

# Core modules (no TensorFlow)
print("\n📦 Checking core modules...")
check("config", "config")
check("core.model_loader", "core/model_loader")
check("core.data_transformer", "core/data_transformer")
check("core.prediction_utils", "core/prediction_utils")
check("components.sidebar", "components/sidebar")
check("components.cards", "components/cards")
check("components.charts", "components/charts")
check("components.theme", "components/theme")
check("pages.landing", "pages/landing")
check("pages.manual_prediction", "pages/manual_prediction")
check("pages.analytics", "pages/analytics")
check("pages.history", "pages/history")
check("pages.settings", "pages/settings")
check("pages.about", "pages/about")
check("pages.encyclopedia", "pages/encyclopedia")
check("utils.history_manager", "utils/history_manager")
check("utils.export_utils", "utils/export_utils")
check("explainability.shap_explainer", "explainability/shap_explainer")

# openpyxl
print("\n📦 Checking optional deps...")
try:
    import openpyxl
    print(f"✅ openpyxl {openpyxl.__version__}")
except ImportError:
    errors.append("openpyxl not installed")
    print("❌ openpyxl not installed")

# Model files
print("\n📁 Checking model files...")
import os
for f in ["models/svm_model.pkl", "models/label_encoders.pkl", "models/scaler.pkl"]:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        errors.append(f"Missing: {f}")
        print(f"❌ {f} MISSING")

# Model loading test
print("\n🧠 Testing model loading...")
try:
    import pickle
    with open("models/svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print(f"✅ SVM model: {type(model).__name__}")
    print(f"✅ Label encoders: {len(label_encoders)} features")
    print(f"✅ Scaler: {type(scaler).__name__}")
except Exception as e:
    errors.append(f"Model loading: {e}")
    print(f"❌ Model loading failed: {e}")

# PDF generation test
print("\n📄 Testing PDF generation...")
try:
    from reports.pdf_generator import generate_safety_report
    from PIL import Image as PILImage
    
    test_image = PILImage.new("RGB", (100, 100), color=(46, 125, 50))
    report_data = {
        "Prediction": "EDIBLE",
        "Confidence": "95.5%",
        "Risk Level": "Safe",
        "Model Used": "SVM",
        "Prediction Time": "0.12 seconds",
        "Disclaimer": "Test disclaimer.",
        "Top Contributing Features": ["odor", "gill-color"],
        "Image": test_image,
    }
    pdf_bytes = generate_safety_report(report_data)
    print(f"✅ PDF generated: {len(pdf_bytes)} bytes")
except Exception as e:
    errors.append(f"PDF generation: {e}")
    print(f"❌ PDF generation failed: {e}")
    traceback.print_exc()

# Export utilities test
print("\n📤 Testing export utilities...")
try:
    from utils.export_utils import export_to_csv, export_to_json
    test_data = [{"timestamp": "2024-01-01", "prediction": "EDIBLE", "confidence": "95%"}]
    print(f"✅ CSV export: {len(export_to_csv(test_data))} bytes")
    print(f"✅ JSON export: {len(export_to_json(test_data))} bytes")
except Exception as e:
    errors.append(f"Export utilities: {e}")
    print(f"❌ Export utilities failed: {e}")

# Chart generation test
print("\n📊 Testing chart generation...")
try:
    from components.charts import confidence_gauge, probability_bar_chart
    import numpy as np
    fig1 = confidence_gauge(85.5, "EDIBLE")
    fig2 = probability_bar_chart(np.array([0.855, 0.145]))
    print("✅ Confidence gauge chart")
    print("✅ Probability bar chart")
except Exception as e:
    errors.append(f"Chart generation: {e}")
    print(f"❌ Chart generation failed: {e}")

# SHAP test
print("\n🤖 Testing SHAP module...")
try:
    from explainability.shap_explainer import get_shap_explanation, plot_shap_bar
    print("✅ SHAP module imports OK")
except Exception as e:
    errors.append(f"SHAP module: {e}")
    print(f"❌ SHAP module failed: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"❌ VERIFICATION COMPLETE - {len(errors)} ERROR(S) FOUND")
    for err in errors:
        print(f"   - {err}")
else:
    print("✅ ALL CHECKS PASSED - Application is ready to run!")
print("=" * 60)
