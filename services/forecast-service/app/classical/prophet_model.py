import numpy as np
import pandas as pd
from prophet import Prophet


def forecast_prophet(train_df: pd.DataFrame, horizon: int) -> np.ndarray:
    # Prophet butuh kolom bernama persis 'ds' (tanggal) dan 'y' (nilai)
    df = train_df.rename(columns={"date": "ds", "daily_expense": "y"})

    model = Prophet(
        yearly_seasonality=False,  # data kita cuma 6 bulan, tidak cukup untuk pola tahunan
        weekly_seasonality=True,   # relevan -- pola belanja akhir pekan vs weekday
        daily_seasonality=False,
    )
    model.fit(df[["ds", "y"]])

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    return np.maximum(forecast["yhat"].values[-horizon:], 0)