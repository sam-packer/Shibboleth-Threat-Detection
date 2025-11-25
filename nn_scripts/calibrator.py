import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

class PlattCalibrator:
    """
    Production-ready Platt scaling (logistic calibration).
    Converts raw NN sigmoid outputs into calibrated probabilities.
    """

    def __init__(self, model: LogisticRegression | None = None):
        self.model = model

    def fit(self, preds: np.ndarray, labels: np.ndarray):
        """
        preds: raw NN outputs (sigmoid)
        labels: ground truth 0/1
        """
        lr = LogisticRegression(max_iter=200)
        lr.fit(preds.reshape(-1, 1), labels)
        self.model = lr

    def transform(self, preds: np.ndarray) -> np.ndarray:
        """
        preds: raw NN outputs array
        returns calibrated probabilities
        """
        return self.model.predict_proba(preds.reshape(-1, 1))[:, 1]

    def save(self, path: str):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str):
        model = joblib.load(path)
        return PlattCalibrator(model)
