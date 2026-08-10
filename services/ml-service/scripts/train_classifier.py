import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.classifier.embed import TextEmbedder
from app.classifier.model import ClassifierHead
from data.categories import CATEGORIES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3


def main():
    print(f"Training on device: {DEVICE}")

    df = pd.read_csv("data/synthetic_transactions.csv")

    # Split berbasis merchant: held-out merchant HANYA muncul di test set,
    # tidak pernah dilihat model saat training. Ini test generalisasi asli.
    held_out_merchants = set()
    for cat_data in CATEGORIES.values():
        held_out_merchants.update(cat_data.get("held_out_test", []))

    is_held_out = df["merchant"].isin(held_out_merchants)
    df_train_pool = df[~is_held_out]
    df_test_heldout = df[is_held_out]

    embedder = TextEmbedder()
    print("Embedding descriptions...")

    X_train_pool = embedder.embed(df_train_pool["description"].tolist())
    X_test_heldout = embedder.embed(df_test_heldout["description"].tolist())

    label_encoder = LabelEncoder()
    label_encoder.fit(df["category"])  # fit di semua kategori supaya konsisten

    y_train_pool = label_encoder.transform(df_train_pool["category"])
    y_test_heldout = label_encoder.transform(df_test_heldout["category"])

    # Dari train_pool, sisihkan sebagian lagi jadi "test biasa" (merchant yang SAMA
    # muncul di train & test) sebagai pembanding
    X_train, X_test_normal, y_train, y_test_normal = train_test_split(
        X_train_pool, y_train_pool, test_size=0.2, random_state=42, stratify=y_train_pool
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    model = ClassifierHead(input_dim=X_train.shape[1], num_classes=len(label_encoder.classes_)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Training classifier head...")
    model.train()
    for epoch in range(EPOCHS):
        permutation = torch.randperm(X_train_t.size(0))
        total_loss = 0.0
        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx = permutation[i : i + BATCH_SIZE]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss:.4f}")

    model.eval()

    def evaluate(X, y, label):
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds = model(X_t).argmax(dim=1).cpu().numpy()
        print(f"\n=== Evaluation: {label} ===")
        print(classification_report(y, preds, target_names=label_encoder.classes_, zero_division=0))

    evaluate(X_test_normal, y_test_normal, "Test set biasa (merchant sudah pernah dilihat)")
    evaluate(X_test_heldout, y_test_heldout, "Held-out merchant (BELUM PERNAH dilihat model)")

    torch.save(model.state_dict(), "app/classifier/model_weights.pt")
    np.save("app/classifier/label_classes.npy", label_encoder.classes_)
    print("\nModel saved.")


if __name__ == "__main__":
    main()