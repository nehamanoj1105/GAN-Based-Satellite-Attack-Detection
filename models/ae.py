import os
import torch
import torch.nn as nn
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "archive/data/data/train")
MODEL_PATH = os.path.join(BASE_DIR, "../models/autoencoder.pth")
THRESHOLD_PATH = os.path.join(BASE_DIR, "../models/threshold.npy")

# -----------------------------
# FORCE FEATURE SIZE
# -----------------------------
TARGET_DIM = 25   # ✅ FIXED

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
# LOAD DATA
# -----------------------------
def load_npy_dataset(folder):
    all_data = []

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            arr = np.load(path)

            # reshape if needed
            if len(arr.shape) > 2:
                arr = arr.reshape(-1, arr.shape[-1])

            # ✅ ONLY TAKE 25-FEATURE FILES
            if arr.shape[1] != TARGET_DIM:
                print(f"Skipping {file} (dim {arr.shape[1]})")
                continue

            all_data.append(arr)

    if len(all_data) == 0:
        raise ValueError("❌ No valid 25-feature data found")

    return np.vstack(all_data)


# -----------------------------
# LOAD TRAIN DATA
# -----------------------------
print("📥 Loading TRAIN dataset...")
train_data = load_npy_dataset(TRAIN_PATH)

print("Train shape:", train_data.shape)

X_train = torch.tensor(train_data.astype(np.float32))

# -----------------------------
# MODEL
# -----------------------------
input_dim = TARGET_DIM
model = Autoencoder(input_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# -----------------------------
# TRAINING
# -----------------------------
epochs = 20

print("🚀 Training...")

for epoch in range(epochs):
    optimizer.zero_grad()

    output = model(X_train)
    loss = criterion(output, X_train)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)

print("✅ Model saved at:", MODEL_PATH)

# -----------------------------
# CALCULATE THRESHOLD
# -----------------------------
with torch.no_grad():
    recon = model(X_train)
    errors = torch.mean((X_train - recon) ** 2, dim=1).numpy()

threshold = np.mean(errors) + 3 * np.std(errors)

# ✅ SAVE THRESHOLD
np.save(THRESHOLD_PATH, threshold)

print("\n🔥 FINAL RESULTS")
print("Max Error:", np.max(errors))
print("Mean Error:", np.mean(errors))
print("Recommended Threshold:", threshold)
print("Saved threshold at:", THRESHOLD_PATH)
