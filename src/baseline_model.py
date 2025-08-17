
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@dataclass
class BaselineLinearModel:
    random_state: int = 42
    model_name: str = "LinearRegression"   # or "RidgeCV"
    model: object = None
    is_trained: bool = False

    def _build_features(self, nn_features: Dict) -> pd.DataFrame:
        X_num = nn_features['numeric'].reset_index(drop=True)
        X_oh  = nn_features['onehot'].reset_index(drop=True)
        X = pd.concat([X_num, X_oh], axis=1)
        return X

    def _make_model(self):
        if self.model_name == "RidgeCV":
            return RidgeCV(alphas=(0.1, 1.0, 10.0), store_cv_values=False)
        return LinearRegression()

    def train_and_evaluate(self, nn_features: Dict, y: pd.Series, test_size: float = 0.2) -> Dict:
        X = self._build_features(nn_features)
        y = y.reset_index(drop=True)

        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=self.random_state)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        self.model = self._make_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
        r2 = r2_score(y_test, y_pred)

        return {
            "model": self.model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "y_test": y_test.values,
            "predictions": y_pred
        }
