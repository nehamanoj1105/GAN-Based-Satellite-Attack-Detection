import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# -----------------------------
# AUTOENCODER
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# -----------------------------
# DETECTOR
# -----------------------------
class AutoencoderDetector:
    def __init__(self, model_path, threshold):
        self.device = torch.device("cpu")
        self.model_path = model_path
        self.threshold = threshold
        self.model = None

    def load_model(self, input_dim):
        self.model = Autoencoder(input_dim)

        if self.model_path:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, df):
        df = df.select_dtypes(include=[np.number])
        return df.values.astype(np.float32)

    def detect(self, df):
        data = self.preprocess(df)

        if self.model is None:
            self.load_model(data.shape[1])

        x = torch.tensor(data).to(self.device)

        with torch.no_grad():
            recon = self.model(x)
            loss = torch.mean((x - recon) ** 2, dim=1)

        scores = loss.cpu().numpy()

        return {
            "max_reconstruction_error": float(np.max(scores)),
            "mean_reconstruction_error": float(np.mean(scores)),
            "threshold": float(self.threshold),
            "is_anomaly": bool(np.max(scores) > self.threshold),
            "scores": scores.tolist()   # ✅ IMPORTANT FIX
        }
