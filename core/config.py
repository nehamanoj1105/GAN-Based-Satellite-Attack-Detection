import os
from pathlib import Path

BASE_DIR = Path(**file**).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "autoencoder.pth"))
SCALER_PATH = os.getenv("SCALER_PATH", str(MODELS_DIR / "scaler.pkl"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.05"))
