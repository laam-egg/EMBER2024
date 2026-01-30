import lightgbm as lgb
import numpy as np

class ModelTrainer:
    def __init__(self, params: dict) -> None:
        self.params = params
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> lgb.Booster:
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            self.params,
            train_data,
            valid_sets=[val_data],
        )

        return model
