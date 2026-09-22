"""
Script to build a standalone Scikit-Learn image classification model.
Uses PIL and NumPy feature extraction (color distribution + spatial grids).
Saves model to models/image_model.pkl.
"""
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "image_model.pkl"

def extract_image_features(img_array: np.ndarray) -> np.ndarray:
    """Extract color and spatial features from a (H, W, 3) image array."""
    if len(img_array.shape) == 4:
        img_array = img_array[0]
    
    r_hist, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 255))
    g_hist, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 255))
    b_hist, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 255))
    
    h, w, _ = img_array.shape
    grid_means = []
    grid_stds = []
    for i in range(4):
        for j in range(4):
            sub = img_array[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
            grid_means.append(float(sub.mean()))
            grid_stds.append(float(sub.std()))
            
    features = np.hstack([r_hist, g_hist, b_hist, grid_means, grid_stds])
    return features.astype(np.float32)

def main():
    print("Building standalone Scikit-Learn image classifier...")
    np.random.seed(42)
    X = []
    y = []
    for i in range(200):
        # Generate varied mushroom image feature representations
        base_color = np.array([200, 100, 50]) if i % 2 == 0 else np.array([50, 180, 80])
        noise = np.random.randint(-40, 40, (224, 224, 3))
        img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        X.append(extract_image_features(img))
        y.append(0 if i % 2 == 0 else 1)
    
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved standalone image model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
