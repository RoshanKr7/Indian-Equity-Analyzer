"""
Temporal Fusion Transformer (TFT) — Inference wrapper.
Training done on Kaggle GPU (kaggle/train_tft.ipynb).

Architecture:
  Input (seq_len, n_features)
    → Input Projection (ip)
    → Variable Selection Network (vsn)
    → BiLSTM Encoder (bilstm) + AddNorm (an1)
    → Interpretable Multi-Head Attention (attn) + AddNorm (an2)
    → Gated Residual Network (ff) + AddNorm (an3)
    → Adaptive Avg Pooling + Classifier (cls) → [P(Sell), P(Hold), P(Buy)]

Output files from kaggle/train_tft.ipynb:
  - models/pretrained/tft_weights.pth
  - models/pretrained/tft_config.json
  - models/pretrained/feature_scaler.pkl
"""

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from config.settings import LSTM_SEQUENCE_LEN, TFT_WEIGHTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks (matching kaggle/train_tft.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

class GatedLinearUnit(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, d * 2)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        o, g = self.fc(x).chunk(2, dim=-1)
        return o * self.gate(g)


class AddNorm(nn.Module):
    def __init__(self, d: int, dr: float = 0.1):
        super().__init__()
        self.glu = GatedLinearUnit(d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dr)

    def forward(self, x, res):
        return self.norm(self.drop(self.glu(x)) + res)


class GatedResidualNetwork(nn.Module):
    def __init__(self, d: int, dh: int = None, dr: float = 0.1):
        super().__init__()
        dh = dh or d
        self.fc1 = nn.Linear(d, dh)
        self.fc2 = nn.Linear(dh, d)
        self.elu = nn.ELU()
        self.an = AddNorm(d, dr)

    def forward(self, x):
        return self.an(self.fc2(self.elu(self.fc1(x))), x)


class VariableSelectionNetworkV2(nn.Module):
    def __init__(self, n: int, d: int, dr: float = 0.1):
        super().__init__()
        self.wn = nn.Sequential(nn.Linear(n, d), nn.ELU(), nn.Dropout(dr), nn.Linear(d, n))
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        w = self.sm(self.wn(x))
        return x * w, w.mean(dim=1)


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d: int, nh: int = 4, dr: float = 0.1):
        super().__init__()
        self.nh = nh
        self.dk = d // nh
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dr)

    def forward(self, x):
        B, T, D = x.shape
        H, dk = self.nh, self.dk
        Q = self.q(x).view(B, T, H, dk).transpose(1, 2)
        K = self.k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.v(x).view(B, T, H, dk).transpose(1, 2)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk), dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out), attn.mean(dim=1)


class TemporalFusionTransformer(nn.Module):
    """
    TFT classifier network — identical to kaggle/train_tft.ipynb.
    """

    def __init__(self, nf: int, d: int = 64, nh: int = 4, nl: int = 2, dr: float = 0.2, nc: int = 3):
        super().__init__()
        self.ip = nn.Linear(nf, d)
        self.vsn = VariableSelectionNetworkV2(d, d, dr)
        self.bilstm = nn.LSTM(d, d // 2, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)
        self.an1 = AddNorm(d, dr)
        self.attn = InterpretableMultiHeadAttention(d, nh, dr)
        self.an2 = AddNorm(d, dr)
        self.ff = GatedResidualNetwork(d, d * 2, dr)
        self.an3 = AddNorm(d, dr)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dr)
        self.cls = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr), nn.Linear(d // 2, nc))

    def forward(self, x):
        B, T, F = x.shape
        x = self.ip(x)
        x, _ = self.vsn(x)
        lo, _ = self.bilstm(x)
        x = self.an1(lo, x)
        ao, _ = self.attn(x)
        x = self.an2(ao, x)
        fo = self.ff(x.reshape(B * T, -1)).reshape(B, T, -1)
        x = self.an3(fo, x)
        p = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.cls(self.drop(p))


# ─────────────────────────────────────────────────────────────────────────────
# Inference Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TFTPredictor:
    """
    TFT inference wrapper — loads Kaggle-trained weights from disk.
    """

    def __init__(self):
        self.name   = "tft"
        self.model  = None
        self.scaler = None
        self.config = None

    def check_weights_exist(self) -> bool:
        return os.path.exists(os.path.join(TFT_WEIGHTS_DIR, "tft_weights.pth"))

    def _load_model(self, n_features: int = 28):
        if self.model is not None:
            return

        weights_path = os.path.join(TFT_WEIGHTS_DIR, "tft_weights.pth")
        config_path  = os.path.join(TFT_WEIGHTS_DIR, "tft_config.json")
        scaler_path  = os.path.join(TFT_WEIGHTS_DIR, "feature_scaler.pkl")

        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
            n_features = self.config.get("n_features", n_features)
            d_model    = self.config.get("d_model", 64)
            n_heads    = self.config.get("n_heads", 4)
            n_lstm     = self.config.get("n_lstm_layers", 2)
            dropout    = self.config.get("dropout", 0.2)
        else:
            d_model, n_heads, n_lstm, dropout = 64, 4, 2, 0.2

        self.model = TemporalFusionTransformer(
            nf=n_features,
            d=d_model, nh=n_heads,
            nl=n_lstm, dr=dropout,
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    def predict(self, features: pd.DataFrame, horizon_days: int) -> dict | None:
        """
        Run TFT inference on the latest sequence.
        Returns None if weights not found (xgboost_temporal covers the slot).
        """
        if not self.check_weights_exist():
            return None

        try:
            self._load_model()
        except Exception:
            return None

        # Align features with the exact feature set used in training
        if self.config and "feature_cols" in self.config and self.config["feature_cols"]:
            expected_cols = self.config["feature_cols"]
            features = features.reindex(columns=expected_cols, fill_value=0.0)
        elif self.config and "n_features" in self.config:
            expected_n = self.config["n_features"]
            if features.shape[1] > expected_n:
                features = features.iloc[:, :expected_n]
            elif features.shape[1] < expected_n:
                return None

        seq_len = self.config.get("seq_len", LSTM_SEQUENCE_LEN) if self.config else LSTM_SEQUENCE_LEN
        if len(features) < seq_len:
            return None

        data = features.iloc[-seq_len:].values.astype(np.float32)
        if self.scaler is not None:
            try:
                data = self.scaler.transform(data)
            except Exception:
                pass
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        tensor = torch.FloatTensor(data).unsqueeze(0)   # (1, seq_len, features)

        with torch.no_grad():
            logits = self.model(tensor)
            proba  = torch.softmax(logits, dim=1).numpy()[0]

        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])
        signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        avg_return = (proba[2] - proba[0]) * 0.05 * (horizon_days / 30)

        return {
            "signal":           signal_map[pred_class],
            "confidence":       round(confidence, 4),
            "predicted_return": round(avg_return, 4),
            "probabilities": {
                "Sell": round(float(proba[0]), 4),
                "Hold": round(float(proba[1]), 4),
                "Buy":  round(float(proba[2]), 4),
            },
        }
