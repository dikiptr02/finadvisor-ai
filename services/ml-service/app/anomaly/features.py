import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class AnomalyFeatureBuilder:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        category_encoded = self.encoder.fit_transform(df[["category"]])
        self.fitted = True
        return self._build_features(df, category_encoded)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Encoder belum di-fit. Panggil fit_transform dulu.")
        category_encoded = self.encoder.transform(df[["category"]])
        return self._build_features(df, category_encoded)

    def _build_features(self, df: pd.DataFrame, category_encoded: np.ndarray) -> np.ndarray:
        # log1p meredam skewness nominal (rentang 5rb - 25juta jadi lebih proporsional)
        log_amount = np.log1p(df["amount"].values).reshape(-1, 1)

        date = pd.to_datetime(df["date"])
        day_of_week = date.dt.dayofweek.values.reshape(-1, 1) / 6.0  # normalisasi ke 0-1

        return np.hstack([log_amount, day_of_week, category_encoded])