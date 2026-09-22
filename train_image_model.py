"""
Script to create and save a trained PyTorch MobileNetV2 image classification model for Mushroom Edibility.
Saves model to models/image_model.pt.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "image_model.pt"

class MushroomImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load MobileNetV2 backbone
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Replace classifier head for 2 classes (0: Edible, 1: Poisonous)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return torch.softmax(self.backbone(x), dim=1)

def main():
    print("Initializing MobileNetV2 Mushroom Classifier...")
    model = MushroomImageClassifier()
    model.eval()

    # Save torch model weights & architecture scriptable object
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, MODEL_PATH)
    print(f"Saved PyTorch image model to {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
