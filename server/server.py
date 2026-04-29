import numpy as np
from core.model import AutoencoderDetector

threshold = float(np.load("models/threshold.npy"))

detector = AutoencoderDetector(
    model_path="models/autoencoder.pth",
    threshold=0.3
)

def run_detection(df):
    return detector.detect(df)
