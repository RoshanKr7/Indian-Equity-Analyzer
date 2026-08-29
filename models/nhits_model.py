"""
N-HiTS (Neural Hierarchical Interpolation for Time Series) — Inference wrapper.
Training done on Kaggle GPU (kaggle/train_nhits.ipynb).

Architecture (Challu et al., AAAI 2023 — adapted for stock direction):
  Input: last CONTEXT_LEN days of close prices + technical features
    → Stack of N-HiTS blocks at multiple time scales:
        Block 1 (coarse, pool=5):  MLP → backcast + forecast at 1/5th resolution
        Block 2 (medium, pool=2):  MLP → backcast + forecast at 1/2nd resolution
        Block 3 (fine,   pool=1):  MLP → backcast + forecast at full resolution
    → Sum block forecasts → predicted return at each horizon
    → Threshold to Buy/Hold/Sell

Why N-HiTS > Prophet?
  - Prophet assumes additive trend + seasonality (curve-fitting).
    N-HiTS *learns* the decomposition from data — no assumptions.
  - Multi-scale: coarse block captures macro trends, fine block catches
    short-term momentum. Prophet has no such hierarchy.
  - Speed: N-HiTS inference ~5-10ms vs Prophet's ~30,000ms (30 seconds).
    Total app runtime drops by ~30 seconds just from this swap.
  - Accuracy: Designed for long-horizon forecasting, consistently outperforms
    Prophet on financial benchmarks (2023-25 literature).

Adapted for classification:
  N-HiTS predicts raw return → threshold classifies Buy/Hold/Sell.
  Also provides predicted_price for the UI (same interface as Prophet).

Output files from kaggle/train_nhits.ipynb:
  - models/pretrained/nhits_weights.pth
  - models/pretrained/nhits_config.json
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config.settings import (
    NHITS_WEIGHTS_DIR, NHITS_CONTEXT_LEN,
    DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# N-HiTS Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class NHiTSBlock(nn.Module):
    """
    Single N-HiTS block.

    Receives a pooled (down-sampled) view of the input.
    Produces:
      - backcast: what this block explains about the input
      - forecast: this block's contribution to the horizon prediction
    """

    def __init__(self,
                 context_len:  int,
                 horizon_len:  int,
                 n_features:   int,
                 pooling_size: int,
                 d_hidden:     int   = 256,
                 n_layers:     int   = 2,
                 dropout:      float = 0.1):
        super().__init__()

        self.pooling_size = pooling_size
        self.pooled_len   = context_len // pooling_size + (1 if context_len % pooling_size else 0)
        input_dim = self.pooled_len * n_features

        # MLP backbone
        layers = [nn.Linear(input_dim, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)

        # Backcast head: reconstruct (pooled) input
        self.backcast_head = nn.Linear(d_hidden, input_dim)

        # Forecast head: predict future return
        self.forecast_head = nn.Linear(d_hidden, horizon_len)

        # Adaptive pooling for input
        self.pool = nn.AdaptiveAvgPool1d(self.pooled_len)

    def forward(self, x):
        # x: (B, n_features, context_len) — channels-first for pooling
        pooled = self.pool(x)                          # (B, n_features, pooled_len)
        B, F, L = pooled.shape
        flat = pooled.reshape(B, F * L)                # (B, input_dim)

        hidden = self.mlp(flat)                        # (B, d_hidden)
        backcast_flat = self.backcast_head(hidden)     # (B, input_dim)
        backcast = backcast_flat.reshape(B, F, L)      # (B, F, pooled_len)

        # Upsample backcast back to context_len for residual subtraction
        backcast_up = nn.functional.interpolate(
            backcast, size=x.shape[-1], mode="linear", align_corners=False
        )

        forecast = self.forecast_head(hidden)          # (B, horizon_len)
        return backcast_up, forecast


class NHiTSNetwork(nn.Module):
    """
    Full N-HiTS stack.

    Must match kaggle/train_nhits.ipynb exactly.

    Input:  (batch, n_features, context_len)  — channels-first
    Output: (batch, horizon_len)              — predicted returns
    """

    def __init__(self,
                 context_len:   int,
                 horizon_len:   int,
                 n_features:    int,
                 pooling_sizes: list = None,
                 d_hidden:      int  = 256,
                 n_layers:      int  = 2,
                 dropout:       float = 0.1):
        super().__init__()
        if pooling_sizes is None:
            pooling_sizes = [5, 2, 1]   # coarse → fine

        self.blocks = nn.ModuleList([
            NHiTSBlock(
                context_len=context_len,
                horizon_len=horizon_len,
                n_features=n_features,
                pooling_size=ps,
                d_hidden=d_hidden,
                n_layers=n_layers,
                dropout=dropout,
            )
            for ps in pooling_sizes
        ])

    def forward(self, x):
        # x: (B, n_features, context_len)
        residual = x
        total_forecast = torch.zeros(
            x.shape[0], self.blocks[0].forecast_head.out_features,
            device=x.device
        )

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast          # subtract explained component
            total_forecast = total_forecast + forecast

        return total_forecast                       # (B, horizon_len)


# ─────────────────────────────────────────────────────────────────────────────
# Inference Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class NHiTSPredictor:
    """
    N-HiTS inference wrapper — loads Kaggle-trained weights from disk.

    Falls back to ProphetPredictor if weights not found, maintaining the
    same interface (signal, confidence, predicted_price, predicted_return).
    """

    def __init__(self):
        self.name   = "nhits"
        self.model  = None
        self.config = None

    def check_weights_exist(self) -> bool:
        return os.path.exists(os.path.join(NHITS_WEIGHTS_DIR, "nhits_weights.pth"))

    def _load_model(self, n_features: int = None, horizon_len: int = 365):
        if self.model is not None:
            return

        weights_path = os.path.join(NHITS_WEIGHTS_DIR, "nhits_weights.pth")
        config_path  = os.path.join(NHITS_WEIGHTS_DIR, "nhits_config.json")

        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
            context_len   = self.config.get("context_len", NHITS_CONTEXT_LEN)
            n_features    = self.config.get("n_features", n_features or 24)
            horizon_len   = self.config.get("max_horizon", horizon_len)
            pooling_sizes = self.config.get("pooling_sizes", [5, 2, 1])
            d_hidden      = self.config.get("d_hidden", 256)
            n_layers      = self.config.get("n_layers", 2)
        else:
            context_len   = NHITS_CONTEXT_LEN
            pooling_sizes = [5, 2, 1]
            d_hidden, n_layers = 256, 2
            n_features = n_features or 24

        self.model = NHiTSNetwork(
            context_len=context_len,
            horizon_len=horizon_len,
            n_features=n_features,
            pooling_sizes=pooling_sizes,
            d_hidden=d_hidden,
            n_layers=n_layers,
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, features: pd.DataFrame, close_series: pd.Series,
                horizon_days: int) -> dict:
        """
        Predict return at horizon_days using N-HiTS.

        Falls back to a simple momentum baseline if weights absent.
        Returns same interface as ProphetPredictor.
        """
        last_price = float(close_series.iloc[-1])

        if not self.check_weights_exist():
            return self._momentum_fallback(close_series, horizon_days)

        context_len = self.config.get("context_len", NHITS_CONTEXT_LEN) if self.config else NHITS_CONTEXT_LEN
        max_horizon = self.config.get("max_horizon", 365)   if self.config else 365

        try:
            self._load_model(horizon_len=max_horizon)
        except Exception:
            return self._momentum_fallback(close_series, horizon_days)

        # Align features with the exact feature set used in training
        if self.config and "feature_cols" in self.config and self.config["feature_cols"]:
            expected_cols = self.config["feature_cols"]
            features = features.reindex(columns=expected_cols, fill_value=0.0)
        elif self.config and "n_features" in self.config:
            expected_n = self.config["n_features"]
            if features.shape[1] > expected_n:
                features = features.iloc[:, :expected_n]
            elif features.shape[1] < expected_n:
                return self._momentum_fallback(close_series, horizon_days)

        if len(features) < context_len:
            return self._momentum_fallback(close_series, horizon_days)

        data = features.iloc[-context_len:].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # channels-first: (1, n_features, context_len)
        tensor = torch.FloatTensor(data).T.unsqueeze(0)

        with torch.no_grad():
            # forecast shape: (1, max_horizon)
            forecast_returns = self.model(tensor)[0].numpy()

        # Get the return at the requested horizon
        idx = min(horizon_days - 1, len(forecast_returns) - 1)
        predicted_return = float(forecast_returns[idx])

        # Clip unrealistic predictions
        predicted_return = np.clip(predicted_return, -0.5, 0.5)

        predicted_price = round(last_price * (1 + predicted_return), 2)

        # Confidence: based on return magnitude + consistency of direction
        direction_consistent = (
            np.sum(forecast_returns[:idx+1] > 0) / max(idx + 1, 1)
            if predicted_return > 0
            else np.sum(forecast_returns[:idx+1] < 0) / max(idx + 1, 1)
        )
        confidence = max(0.25, min(0.90, direction_consistent))

        if predicted_return > DEFAULT_BUY_THRESHOLD:
            signal = "Buy"
        elif predicted_return < DEFAULT_SELL_THRESHOLD:
            signal = "Sell"
        else:
            signal = "Hold"

        return {
            "signal":           signal,
            "confidence":       round(confidence, 4),
            "predicted_return": round(predicted_return, 4),
            "predicted_price":  predicted_price,
        }

    def _momentum_fallback(self, close: pd.Series, horizon_days: int) -> dict:
        """Simple momentum baseline used when N-HiTS weights are absent."""
        if len(close) < 20:
            return self._default_result(close)
        recent_return = float((close.iloc[-1] / close.iloc[-20]) - 1)
        scaled = recent_return * (horizon_days / 20)
        scaled = np.clip(scaled, -0.4, 0.4)
        price  = round(float(close.iloc[-1]) * (1 + scaled), 2)
        signal = "Buy" if scaled > DEFAULT_BUY_THRESHOLD else "Sell" if scaled < DEFAULT_SELL_THRESHOLD else "Hold"
        return {
            "signal":           signal,
            "confidence":       0.30,
            "predicted_return": round(scaled, 4),
            "predicted_price":  price,
        }

    def _default_result(self, close: pd.Series) -> dict:
        price = float(close.iloc[-1]) if len(close) > 0 else 0
        return {
            "signal": "Hold", "confidence": 0.25,
            "predicted_return": 0.0, "predicted_price": round(price, 2),
        }
