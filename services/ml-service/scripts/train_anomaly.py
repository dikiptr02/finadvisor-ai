import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn

from app.anomaly.autoencoder_model import Autoencoder
from app.anomaly.features import AnomalyFeatureBuilder
from app.anomaly.isolation_forest_model import IsolationForestDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE_EPOCHS = 50
AE_LR = 1e-3
CONTAMINATION = 0.02


def train_autoencoder(X_train: np.ndarray, input_dim: int) -> Autoencoder:
    model = Autoencoder(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

    model.train()
    for epoch in range(AE_EPOCHS):
        optimizer.zero_grad()
        reconstructed = model(X_train_t)
        loss = criterion(reconstructed, X_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Autoencoder epoch {epoch + 1}/{AE_EPOCHS} - loss: {loss.item():.6f}")

    return model


def autoencoder_anomaly_scores(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        reconstructed = model(X_t)
        errors = torch.mean((X_t - reconstructed) ** 2, dim=1)
    return errors.cpu().numpy()


def evaluate(scores: np.ndarray, y_true: np.ndarray, method_name: str, contamination: float = CONTAMINATION):
    # threshold: ambil top-N% skor tertinggi sebagai "anomali", N% = contamination
    # (tanpa pakai label asli untuk menentukan threshold -- ini murni unsupervised)
    threshold = np.quantile(scores, 1 - contamination)
    y_pred = scores >= threshold

    auc = roc_auc_score(y_true, scores)

    print(f"\n=== {method_name} ===")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["normal", "anomali"], zero_division=0))


def main():
    print(f"Device: {DEVICE}")

    df = pd.read_csv("shared_data/synthetic_transactions.csv")
    y_true = df["is_anomaly_ground_truth"].values

    feature_builder = AnomalyFeatureBuilder()
    X = feature_builder.fit_transform(df)
    print(f"Feature shape: {X.shape}")

    # --- Isolation Forest ---
    print("\nTraining Isolation Forest...")
    iso_forest = IsolationForestDetector(contamination=CONTAMINATION)
    iso_forest.fit(X)
    iso_scores = iso_forest.anomaly_score(X)
    evaluate(iso_scores, y_true, "Isolation Forest")

    # --- Autoencoder ---
    print("\nTraining Autoencoder...")
    ae_model = train_autoencoder(X, input_dim=X.shape[1])
    ae_scores = autoencoder_anomaly_scores(ae_model, X)
    evaluate(ae_scores, y_true, "Autoencoder")

    print("\n=== Studi Komparasi Selesai ===")
    print("Bandingkan ROC-AUC di atas -- semakin mendekati 1.0, semakin baik model")
    print("membedakan anomali dari transaksi normal tanpa pernah diberi tahu label aslinya.")


if __name__ == "__main__":
    main()