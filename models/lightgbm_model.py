"""
LightGBM classifier for stock direction prediction.

Why LightGBM alongside XGBoost?
  - Leaf-wise tree growth captures different split patterns than
    XGBoost's depth-wise trees → stacking reduces variance.
  - DART (Dropouts meet Multiple Additive Regression Trees) boosting:
    randomly drops trees during training, like dropout in neural networks.
    This prevents individual trees from becoming overconfident on noisy
    financial data and dramatically reduces overfitting.
  - Extremely fast training even on 10 years of daily data.
  - `feature_fraction` provides implicit feature selection per tree,
    acting as a second layer of regularisation.

Together, XGBoost + LightGBM form a "gradient boosting committee"
that is standard practice in Kaggle competitions and quant finance.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight

from config.settings import WF_MIN_TRAIN_DAYS, WF_PURGE_DAYS


class LightGBMPredictor:
    """Train-and-predict LightGBM with DART boosting and walk-forward validation."""

    def __init__(self):
        self.name = "lightgbm"

    def predict(self, features: pd.DataFrame, target: pd.Series,
                horizon_days: int) -> dict:
        mask = target.notna()
        X = features.loc[mask].copy()
        y = target.loc[mask].astype(int).copy()

        if len(X) < WF_MIN_TRAIN_DAYS + horizon_days:
            return self._default_result()

        purge = horizon_days + WF_PURGE_DAYS
        train_end = len(X) - purge
        if train_end < WF_MIN_TRAIN_DAYS:
            return self._default_result()

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_pred = features.iloc[[-1]]

        # Per-sample class weights for balanced training
        sample_weights = compute_sample_weight("balanced", y_train)

        model = lgb.LGBMClassifier(
            boosting_type="dart",       # Dropout trees — best regularisation for finance
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,              # Conservative — reduces overfit vs default 127
            min_child_samples=40,       # Larger min-leaf for noisy financial data
            feature_fraction=0.7,       # Sample 70% of features per tree (implicit selection)
            bagging_fraction=0.8,       # Row subsampling per iteration
            bagging_freq=1,
            reg_alpha=0.3,              # L1 regularisation
            reg_lambda=2.0,             # L2 regularisation
            drop_rate=0.1,              # DART: drop 10% of trees per round
            skip_drop=0.5,              # DART: 50% chance to skip dropout
            random_state=42,
            verbose=-1,
            objective="multiclass",
            num_class=3,
            n_jobs=-1,                  # Use all CPU cores
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)

        proba = model.predict_proba(X_pred)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        avg_return = (proba[2] - proba[0]) * 0.05 * (horizon_days / 30)

        return {
            "signal": signal_map[pred_class],
            "confidence": round(confidence, 4),
            "predicted_return": round(avg_return, 4),
            "probabilities": {
                "Sell": round(float(proba[0]), 4),
                "Hold": round(float(proba[1]), 4),
                "Buy":  round(float(proba[2]), 4),
            },
        }

    def _default_result(self):
        return {
            "signal": "Hold",
            "confidence": 0.33,
            "predicted_return": 0.0,
            "probabilities": {"Sell": 0.33, "Hold": 0.34, "Buy": 0.33},
        }
