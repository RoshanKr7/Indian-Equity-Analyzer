"""
XGBoost Temporal — sliding-window sequential predictor (local LSTM replacement).

Why this instead of LSTM for local inference?
  - LSTM requires pretrained weights (GPU-trained on Kaggle) — a barrier.
  - XGBoost Temporal flattens a 60-day rolling window of features into a wide
    vector and trains XGBoost on it. This teaches XGBoost *temporal order*:
    features from day T-59 are treated differently from day T-0 because they
    occupy different columns in the feature matrix.
  - Captures short-term momentum, regime shifts, and sequence-dependent patterns
    that standard (non-windowed) gradient boosters cannot.
  - Zero setup — trains in ~3-5 seconds on CPU, no weights file needed.
  - When the Kaggle-trained BiLSTM weights exist, the LSTM model (lstm_model.py)
    takes over. This model provides a reliable fallback / complementary signal.

Architecture:
  Input: last SEQ_LEN rows of features (60 × N_features)
  → Flatten → wide feature vector (60 × N_features dimensions)
  → XGBoost multi-class classifier
  → [P(Sell), P(Hold), P(Buy)]
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config.settings import (
    WF_MIN_TRAIN_DAYS, WF_PURGE_DAYS,
    XGBT_SEQUENCE_LEN, XGBT_N_ESTIMATORS,
)


class XGBoostTemporalPredictor:
    """Sliding-window XGBoost that captures temporal patterns like an LSTM."""

    def __init__(self):
        self.name = "xgboost_temporal"

    def predict(self, features: pd.DataFrame, target: pd.Series,
                horizon_days: int) -> dict:
        """
        Build windowed feature matrix and train/predict with XGBoost.

        Returns dict with signal, confidence, predicted_return.
        """
        seq_len = XGBT_SEQUENCE_LEN
        mask = target.notna()
        X_raw = features.copy()
        y_raw = target.copy()

        n = len(X_raw)
        if n < seq_len + WF_MIN_TRAIN_DAYS + horizon_days:
            return self._default_result()

        # Build windowed samples: each sample = concat of last seq_len rows
        # Only build samples where we have seq_len history
        Xw_list = []
        y_list   = []

        for i in range(seq_len, n):
            # Skip if target at i is NaN (future unknown at time of training)
            lbl = y_raw.iloc[i]
            if pd.isna(lbl):
                continue
            window = X_raw.iloc[i - seq_len:i].values.flatten()
            Xw_list.append(window)
            y_list.append(int(lbl))

        if len(Xw_list) < WF_MIN_TRAIN_DAYS:
            return self._default_result()

        Xw = np.array(Xw_list, dtype=np.float32)
        yw = np.array(y_list, dtype=np.int32)

        # Walk-forward: train on all but last (horizon + purge) samples
        purge = horizon_days + WF_PURGE_DAYS
        train_end = len(Xw) - purge
        if train_end < WF_MIN_TRAIN_DAYS:
            return self._default_result()

        X_train = Xw[:train_end]
        y_train = yw[:train_end]

        # Prediction point: last seq_len rows of full features (may include NaN-target rows)
        if len(X_raw) < seq_len:
            return self._default_result()
        X_pred = X_raw.iloc[-seq_len:].values.flatten().reshape(1, -1)

        # Replace NaNs/Infs that might have crept in
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_pred  = np.nan_to_num(X_pred,  nan=0.0, posinf=0.0, neginf=0.0)

        model = XGBClassifier(
            n_estimators=XGBT_N_ESTIMATORS,
            max_depth=4,            # Shallow — wide feature vector, risk of overfit
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.4,   # Low — wide feature space, random subsampling
            reg_alpha=1.0,          # Strong L1 to kill redundant lag features
            reg_lambda=2.0,
            min_child_weight=20,    # Each leaf needs substantial support
            tree_method="hist",
            device="cpu",
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )

        try:
            model.fit(X_train, y_train)
            raw_proba = model.predict_proba(X_pred)[0]

            proba = [0.0, 0.0, 0.0]
            if hasattr(model, "classes_"):
                for idx, cls in enumerate(model.classes_):
                    if 0 <= cls < 3:
                        proba[cls] = float(raw_proba[idx])
            else:
                proba = list(raw_proba)

            pred_class = int(np.argmax(proba))
            confidence = float(proba[pred_class]) if sum(proba) > 0 else 0.33

            signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
            avg_return = (proba[2] - proba[0]) * 0.05 * (horizon_days / 30)

            return {
                "signal": signal_map.get(pred_class, "Hold"),
                "confidence": round(confidence, 4),
                "predicted_return": round(avg_return, 4),
                "probabilities": {
                    "Sell": round(float(proba[0]), 4),
                    "Hold": round(float(proba[1]), 4),
                    "Buy":  round(float(proba[2]), 4),
                },
            }
        except Exception:
            return self._default_result()

    def _default_result(self):
        return {
            "signal": "Hold",
            "confidence": 0.33,
            "predicted_return": 0.0,
            "probabilities": {"Sell": 0.33, "Hold": 0.34, "Buy": 0.33},
        }
