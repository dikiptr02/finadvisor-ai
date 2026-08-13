import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from app.classical.arima_model import forecast_arima
from app.classical.prophet_model import forecast_prophet
from app.deep_learning.lstm_model import ForecastLSTM, build_windowed_dataset, recursive_forecast
from data.build_timeseries import build_daily_expense_series

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLE_USERS = 30
HORIZON = 14  # forecast 14 hari ke depan
WINDOW = 14   # LSTM lihat 14 hari terakhir untuk prediksi 1 hari berikutnya
LSTM_EPOCHS = 30


def train_global_lstm(daily_df: pd.DataFrame, sample_user_ids: list[str]) -> ForecastLSTM:
    # LSTM dilatih dari SEMUA user (bukan hanya sample) supaya global model
    # benar-benar belajar dari sebanyak mungkin pola, lalu dites khusus di sample.
    all_X, all_y = [], []
    for user_id, group in daily_df.groupby("user_id"):
        series = group.sort_values("date")["daily_expense"].values
        train_series = series[:-HORIZON] if len(series) > HORIZON else series
        if len(train_series) <= WINDOW:
            continue
        X, y = build_windowed_dataset(train_series, WINDOW)
        all_X.append(X)
        all_y.append(y)

    X_train = np.concatenate(all_X)
    y_train = np.concatenate(all_y)

    X_t = torch.tensor(X_train, dtype=torch.float32).reshape(-1, WINDOW, 1).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    model = ForecastLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Training global LSTM di {len(X_train)} window dari semua user...")
    model.train()
    for epoch in range(LSTM_EPOCHS):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{LSTM_EPOCHS} - loss: {loss.item():.2f}")

    return model


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    # Tambah epsilon kecil untuk hindari div-by-zero di MAPE saat expense harian = 0
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    mape = mean_absolute_percentage_error(y_true_safe, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, rmse


def main():
    daily_df = build_daily_expense_series("shared_data/synthetic_transactions.csv")

    all_users = daily_df["user_id"].unique()
    rng = np.random.default_rng(42)
    sample_users = rng.choice(all_users, size=min(N_SAMPLE_USERS, len(all_users)), replace=False)

    lstm_model = train_global_lstm(daily_df, sample_users)

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
            mape, rmse = evaluate_forecast(y_true, arima_pred)
            results["arima"].append({"mape": mape, "rmse": rmse})
        except Exception as exc:
            print(f"  ARIMA gagal: {exc}")

        try:
            prophet_pred = forecast_prophet(train_df[["date", "daily_expense"]], HORIZON)
            mape, rmse = evaluate_forecast(y_true, prophet_pred)
            results["prophet"].append({"mape": mape, "rmse": rmse})
        except Exception as exc:
            print(f"  Prophet gagal: {exc}")

        try:
            last_window = train_df["daily_expense"].values[-WINDOW:]
            lstm_pred = recursive_forecast(lstm_model, last_window, HORIZON, DEVICE)
            mape, rmse = evaluate_forecast(y_true, lstm_pred)
            results["lstm"].append({"mape": mape, "rmse": rmse})
        except Exception as exc:
            print(f"  LSTM gagal: {exc}")

    print("\n" + "=" * 50)
    print("STUDI KOMPARASI FORECASTING")
    print("=" * 50)
    for method, scores in results.items():
        if not scores:
            print(f"{method}: tidak ada hasil valid")
            continue
        avg_mape = np.mean([s["mape"] for s in scores])
        avg_rmse = np.mean([s["rmse"] for s in scores])
        print(f"{method.upper():10s} | avg MAPE: {avg_mape:.2%} | avg RMSE: {avg_rmse:,.0f} | n_users: {len(scores)}")


if __name__ == "__main__":
    main()