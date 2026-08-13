import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def forecast_arima(train_series: pd.Series, horizon: int) -> np.ndarray:
    # Order (1,1,1) dipakai sebagai konfigurasi sederhana/tetap -- bukan hasil
    # auto-tuning per user. Ini simplifikasi yang disengaja untuk studi komparasi
    # arsitektur; di production biasanya order di-tuning per series (misal via
    # auto_arima) yang lebih mahal secara komputasi.
    model = ARIMA(train_series, order=(1, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)
    return np.maximum(forecast.values, 0)  # pengeluaran tidak boleh negatif