import numpy as np
import torch
import torch.nn as nn


class ForecastLSTM(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # ambil output timestep terakhir
        return self.fc(last_hidden)


def build_windowed_dataset(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def recursive_forecast(model: ForecastLSTM, last_window: np.ndarray, horizon: int, device: torch.device) -> np.ndarray:
    model.eval()
    window = list(last_window)
    predictions = []

    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(window[-len(last_window):], dtype=torch.float32).reshape(1, -1, 1).to(device)
            pred = model(x).item()
            pred = max(pred, 0)
            predictions.append(pred)
            window.append(pred)  # hasil prediksi dipakai sebagai input step berikutnya

    return np.array(predictions)