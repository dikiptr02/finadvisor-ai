import numpy as np
import torch
from torch import nn


class ForecastLSTM(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class SeriesScaler:
    """Scaler untuk deret waktu expense harian: log1p + standarisasi (mean/std).

    Kenapa log1p diperlukan: expense harian sangat heavy-tailed karena anomali yang
    sengaja ditanam (nominal 5-15x lipat kebiasaan). Standarisasi mean/std biasa SANGAT
    sensitif terhadap outlier ekstrem seperti ini -- beberapa hari anomali membuat std
    membengkak, sehingga semua hari normal "terkompres" mendekati 0 setelah dinormalisasi,
    dan LSTM gagal belajar pola apa pun (collapse ke prediksi rata-rata/nol).
    log1p meredam pengaruh nilai ekstrem SEBELUM standarisasi dihitung, sama seperti
    solusi yang sudah kita pakai di fitur anomaly detection (Fase 5).
    """

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, values: np.ndarray) -> "SeriesScaler":
        log_values = np.log1p(values)
        self.mean = float(np.mean(log_values))
        self.std = float(np.std(log_values)) if np.std(log_values) > 1e-8 else 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        log_values = np.log1p(values)
        return (log_values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        log_values = values * self.std + self.mean
        return np.expm1(log_values)


def build_windowed_dataset(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def recursive_forecast(
    model: ForecastLSTM,
    last_window_scaled: np.ndarray,
    horizon: int,
    device: torch.device,
    scaler: SeriesScaler,
) -> np.ndarray:
    model.eval()
    window = list(last_window_scaled)
    predictions_scaled = []

    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(window[-len(last_window_scaled):], dtype=torch.float32).reshape(1, -1, 1).to(device)
            pred_scaled = model(x).item()
            predictions_scaled.append(pred_scaled)
            window.append(pred_scaled)

    predictions = scaler.inverse_transform(np.array(predictions_scaled))
    return np.maximum(predictions, 0)