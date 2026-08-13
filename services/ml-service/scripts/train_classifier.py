import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn

from app.classifier.embed import TextEmbedder
from app.classifier.model import ClassifierHead
from data.categories import CATEGORIES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
EARLY_STOPPING_PATIENCE = 5
MODEL_WEIGHTS_PATH = "app/classifier/model_weights.pt"
LABEL_CLASSES_PATH = "app/classifier/label_classes.npy"


def get_held_out_merchants() -> set[str]:
    held_out = set()
    for cat_data in CATEGORIES.values():
        held_out.update(cat_data.get("held_out_test", []))
    return held_out


def evaluate(model, X, y, label_encoder, label: str):
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    model.train()

    print(f"\n=== Evaluation: {label} ===")
    print(classification_report(y, preds, target_names=label_encoder.classes_, zero_division=0))


def compute_val_loss(model, criterion, X_val_t, y_val_t) -> float:
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
    model.train()
    return val_loss


def main():
    print(f"Training on device: {DEVICE}")

    df = pd.read_csv("shared_data/synthetic_transactions.csv")

    held_out_merchants = get_held_out_merchants()
    is_held_out = df["merchant"].isin(held_out_merchants)

    df_pool = df[~is_held_out].reset_index(drop=True)
    df_heldout = df[is_held_out].reset_index(drop=True)

    print(f"Total data: {len(df)} | Pool (train/val/test): {len(df_pool)} | Held-out: {len(df_heldout)}")

    embedder = TextEmbedder()
    print("Embedding descriptions...")
    X_pool = embedder.embed(df_pool["description"].tolist())
    X_heldout = embedder.embed(df_heldout["description"].tolist())

    label_encoder = LabelEncoder()
    label_encoder.fit(df["category"])

    y_pool = label_encoder.transform(df_pool["category"])
    y_heldout = label_encoder.transform(df_heldout["category"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_pool, y_pool, test_size=0.30, random_state=42, stratify=y_pool
    )
    X_val, X_test_normal, y_val, y_test_normal = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test normal: {len(X_test_normal)} | Held-out: {len(X_heldout)}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    model = ClassifierHead(input_dim=X_train.shape[1], num_classes=len(label_encoder.classes_)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining classifier head...")
    best_val_loss = float("inf")
    patience_counter = 0

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

        val_loss = compute_val_loss(model, criterion, X_val_t, y_val_t)
        print(f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {total_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping di epoch {epoch + 1} (val_loss tidak membaik selama {EARLY_STOPPING_PATIENCE} epoch)")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))

    evaluate(model, X_test_normal, y_test_normal, label_encoder, "Test set biasa (merchant sudah pernah dilihat)")
    evaluate(model, X_heldout, y_heldout, label_encoder, "Held-out merchant (BELUM PERNAH dilihat model)")

    np.save(LABEL_CLASSES_PATH, label_encoder.classes_)
    print(f"\nModel checkpoint terbaik tersimpan di {MODEL_WEIGHTS_PATH}")
    print(f"Label classes tersimpan di {LABEL_CLASSES_PATH}")


if __name__ == "__main__":
    main()