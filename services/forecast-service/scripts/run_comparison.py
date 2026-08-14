import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from torch import nn

from app.classical.arima_model import forecast_arima
from app.classical.prophet_model import forecast_prophet
from app.deep_learning.lstm_model import (
    ForecastLSTM,
    SeriesScaler,
    build_windowed_dataset,
    recursive_forecast,
)
from data.build_timeseries import build_daily_expense_series

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLE_USERS = 30
HORIZON = 14  # forecast 14 hari ke depan
WINDOW = 14   # LSTM lihat 14 hari terakhir untuk prediksi 1 hari berikutnya
LSTM_EPOCHS = 150


def train_global_lstm(daily_df: pd.DataFrame) -> tuple[ForecastLSTM, SeriesScaler]:
    global_scaler = SeriesScaler().fit(daily_df["daily_expense"].values)

    all_X, all_y = [], []
    for _user_id, group in daily_df.groupby("user_id"):
        series = group.sort_values("date")["daily_expense"].values
        train_series = series[:-HORIZON] if len(series) > HORIZON else series
        if len(train_series) <= WINDOW:
            continue

        train_series_scaled = global_scaler.transform(train_series)
        X, y = build_windowed_dataset(train_series_scaled, WINDOW)
        all_X.append(X)
        all_y.append(y)

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)

    # Train/validation split (90/10) untuk early stopping
    n = len(X_all)
    rng_split = np.random.default_rng(42)
    indices = rng_split.permutation(n)
    n_val = max(1, int(n * 0.1))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_tr = torch.tensor(X_all[train_idx], dtype=torch.float32).reshape(-1, WINDOW, 1).to(DEVICE)
    y_tr = torch.tensor(y_all[train_idx], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    X_va = torch.tensor(X_all[val_idx], dtype=torch.float32).reshape(-1, WINDOW, 1).to(DEVICE)
    y_va = torch.tensor(y_all[val_idx], dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    model = ForecastLSTM(hidden_dim=64, num_layers=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    print(f"Training global LSTM di {n} window (train={len(train_idx)}, val={len(val_idx)})...")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience_limit = 20

    for epoch in range(LSTM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = criterion(val_pred, y_va).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1}/{LSTM_EPOCHS}"
                f" - train_loss: {loss.item():.4f}"
                f" - val_loss: {val_loss:.4f}"
                f" - lr: {current_lr:.1e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  Early stopping di epoch {epoch + 1} (val_loss terbaik: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, global_scaler


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE klasik TIDAK BISA dipakai apa adanya di sini -- banyak hari expense = 0,
    # dan pembagian dengan angka mendekati 0 menghasilkan persentase yang meledak
    # tak berarti (ini bug di versi sebelumnya). Solusi: hitung MAPE HANYA di hari
    # dengan expense > 0.
    nonzero_mask = y_true > 0
    if nonzero_mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[nonzero_mask], y_pred[nonzero_mask])
    else:
        mape = float("nan")

    # WAPE (Weighted Absolute Percentage Error) -- metrik agregat yang lebih stabil
    # untuk deret waktu dengan banyak nol, karena membagi TOTAL error dengan TOTAL
    # actual (bukan per-titik), sehingga tidak meledak akibat pembagian per hari.
    total_actual = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / total_actual if total_actual > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "mape": mape, "wape": wape}


def main():
    daily_df = build_daily_expense_series("shared_data/synthetic_transactions.csv")

    all_users = daily_df["user_id"].unique()
    rng = np.random.default_rng(42)
    sample_users = rng.choice(all_users, size=min(N_SAMPLE_USERS, len(all_users)), replace=False)

    lstm_model, scaler = train_global_lstm(daily_df)

    results = {"arima": [], "prophet": [], "lstm": []}

    for i, user_id in enumerate(sample_users):
        group = daily_df[daily_df["user_id"] == user_id].sort_values("date").reset_index(drop=True)
        if len(group) <= HORIZON + WINDOW:
            continue

        train_df = group.iloc[:-HORIZON]
        test_df = group.iloc[-HORIZON:]
        y_true = test_df["daily_expense"].values

        print(f"\n[{i + 1}/{len(sample_users)}] User {user_id[:8]}...")

        try:
            arima_pred = forecast_arima(train_df["daily_expense"], HORIZON)
            results["arima"].append(evaluate_forecast(y_true, arima_pred))
        except Exception as exc:  # noqa: BLE001
            print(f"  ARIMA gagal: {exc}")

        try:
            prophet_pred = forecast_prophet(train_df[["date", "daily_expense"]], HORIZON)
            results["prophet"].append(evaluate_forecast(y_true, prophet_pred))
        except Exception as exc:  # noqa: BLE001
            print(f"  Prophet gagal: {exc}")

        try:
            last_window_raw = train_df["daily_expense"].values[-WINDOW:]
            last_window_scaled = scaler.transform(last_window_raw)
            lstm_pred = recursive_forecast(lstm_model, last_window_scaled, HORIZON, DEVICE, scaler)
            results["lstm"].append(evaluate_forecast(y_true, lstm_pred))
        except Exception as exc:  # noqa: BLE001
            print(f"  LSTM gagal: {exc}")

    print("\n" + "=" * 70)
    print("STUDI KOMPARASI FORECASTING")
    print("=" * 70)
    for method, scores in results.items():
        if not scores:
            print(f"{method}: tidak ada hasil valid")
            continue
        avg_rmse = np.mean([s["rmse"] for s in scores])
        avg_mae = np.mean([s["mae"] for s in scores])
        avg_mape = np.nanmean([s["mape"] for s in scores])
        avg_wape = np.nanmean([s["wape"] for s in scores])
        print(
            f"{method.upper():10s} | RMSE: {avg_rmse:12,.0f} | MAE: {avg_mae:12,.0f} "
            f"| MAPE (hari>0): {avg_mape:.2%} | WAPE: {avg_wape:.2%} | n_users: {len(scores)}"
        )


if __name__ == "__main__":
    main()