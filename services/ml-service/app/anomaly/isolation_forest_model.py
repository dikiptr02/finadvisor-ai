import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        # contamination = perkiraan proporsi anomali, kita samakan dengan yang kita
        # tanam di generator (2%) -- di dunia nyata ini biasanya estimasi kasar dari
        # domain expert, bukan angka yang pasti diketahui
        self.model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)

    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn convention: -1 = anomali, 1 = normal. Kita ubah ke boolean supaya
        # konsisten dengan ground truth kita (True = anomali)
        raw_pred = self.model.predict(X)
        return raw_pred == -1

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # semakin rendah/negatif, semakin dianggap anomali
        return -self.model.score_samples(X)