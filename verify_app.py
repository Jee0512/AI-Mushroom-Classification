"""
Temporary verification script to ensure the redesigned app loads correctly.
"""
import sys
import traceback

print("=" * 60)
print("🍄 Mushroom Classifier - Verification Script")
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

# 1. Check imports for each module
print("\n📦 Checking module imports...")
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
check("pages.image_prediction", "pages/image_prediction")
check("pages.analytics", "pages/analytics")
check("pages.history", "pages/history")
check("pages.settings", "pages/settings")
check("pages.about", "pages/about")
check("pages.encyclopedia", "pages/encyclopedia")
check("utils.history_manager", "utils/history_manager")
check("utils.export_utils", "utils/export_utils")
check("explainability.shap_explainer", "explainability/shap_explainer")
check("explainability.grad_cam_explainer", "explainability/grad_cam_explainer")
check("vision.image_processor", "vision/image_processor")
check("vision.image_classifier", "vision/image_classifier")
check("reports.pdf_generator", "reports/pdf_generator")

# 2. Check openpyxl (for Excel export)
print("\n📦 Checking optional dependencies...")
try:
    import openpyxl
    print(f"✅ openpyxl {openpyxl.__version__}")
except ImportError:
    errors.append("openpyxl not installed (needed for Excel export)")
    print("❌ openpyxl not installed (needed for Excel export)")

# 3. Verify model files
print("\n📁 Checking model files...")
import os
for f in ["models/svm_model.pkl", "models/label_encoders.pkl", "models/scaler.pkl"]:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        errors.append(f"Missing model file: {f}")
        print(f"❌ {f} MISSING")

# 4. Verify dataset
print("\n📁 Checking dataset...")
if os.path.exists("data/mushrooms.csv"):
    print("✅ data/mushrooms.csv")
else:
    errors.append("data/mushrooms.csv not found")
    print("❌ data/mushrooms.csv NOT FOUND")

# 5. Verify model loading works
print("\n🧠 Testing model loading...")
try:
    import pickle
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.svm import SVC
    
    with open("models/svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    print(f"✅ SVM model: {type(model).__name__}")
    print(f"✅ Label encoders: {len(label_encoders)} features")
    print(f"✅ Scaler: {type(scaler).__name__}")
    
    # Test feature options
    feature_names = list(label_encoders.keys())
    print(f"✅ Features: {feature_names}")
    
except Exception as e:
    errors.append(f"Model loading test: {e}")
    print(f"❌ Model loading failed: {e}")

# 6. Test PDF generation
print("\n📄 Testing PDF generation...")
try:
    from reports.pdf_generator import generate_safety_report
    from PIL import Image as PILImage
    import io
    
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
    errors.append(f"PDF generation test: {e}")
    print(f"❌ PDF generation failed: {e}")
    traceback.print_exc()

# 7. Test export utilities
print("\n📤 Testing export utilities...")
try:
    from utils.export_utils import export_to_csv, export_to_json
    test_data = [{"timestamp": "2024-01-01", "prediction": "EDIBLE", "confidence": "95%"}]
    csv_bytes = export_to_csv(test_data)
    json_bytes = export_to_json(test_data)
    print(f"✅ CSV export: {len(csv_bytes)} bytes")
    print(f"✅ JSON export: {len(json_bytes)} bytes")
except Exception as e:
    errors.append(f"Export utilities test: {e}")
    print(f"❌ Export utilities failed: {e}")

# 8. Test chart generation
print("\n📊 Testing chart generation...")
try:
    from components.charts import confidence_gauge, probability_bar_chart
    import numpy as np
    
    fig1 = confidence_gauge(85.5, "EDIBLE")
    fig2 = probability_bar_chart(np.array([0.855, 0.145]))
    print("✅ Confidence gauge chart")
    print("✅ Probability bar chart")
except Exception as e:
    errors.append(f"Chart generation test: {e}")
    print(f"❌ Chart generation failed: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"❌ VERIFICATION COMPLETE - {len(errors)} ERROR(S) FOUND")
    for err in errors:
        print(f"   - {err}")
else:
    print("✅ ALL CHECKS PASSED - Application is ready to run!")
print("=" * 60)
