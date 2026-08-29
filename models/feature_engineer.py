"""
Central feature engineering pipeline.

Every model in the ensemble uses features built here, ensuring
consistency. The pipeline produces:
  - Price-derived features (returns, volatility, drawdown)
  - Technical indicator values
  - Market context features
  - Target variable (Buy=2, Hold=1, Sell=0)
"""

import numpy as np
import pandas as pd

from config.settings import DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD


def build_features(
    stock_df: pd.DataFrame,
    market_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from OHLCV + market context.

    Expects *stock_df* to already have technical indicator columns
    (from ``technical_analysis.compute_all_indicators``).
    """
    feat = pd.DataFrame(index=stock_df.index)

    c = stock_df["Close"]

    # ── Price-derived ──
    for w in [1, 5, 10, 20, 60]:
        feat[f"ret_{w}d"] = np.log(c / c.shift(w))
    for w in [10, 20, 60]:
        feat[f"vol_{w}d"] = c.pct_change().rolling(w).std()

    feat["price_vs_sma50"]  = stock_df.get("Price_vs_SMA50", 0)
    feat["price_vs_sma200"] = stock_df.get("Price_vs_SMA200", 0)
    feat["drawdown_52w"]    = stock_df.get("Drawdown_52w", 0)

    # ── Technical indicators ──
    tech_cols = [
        "RSI_14", "StochRSI", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_PctB", "ATR_14", "ADX_14", "Williams_R", "CCI_20",
        "Volume_Ratio", "Volume_Trend", "OBV",
    ]
    for col in tech_cols:
        if col in stock_df.columns:
            feat[col] = stock_df[col]

    # Normalise OBV to rate-of-change
    if "OBV" in feat.columns:
        feat["OBV_roc"] = feat["OBV"].pct_change(20)
        feat.drop(columns=["OBV"], inplace=True)

    # ── Market context ──
    if market_features is not None:
        for col in market_features.columns:
            feat[col] = market_features[col].reindex(feat.index)

    # Clean up
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.ffill(inplace=True)
    feat.fillna(0, inplace=True)

    return feat


def build_target(
    close: pd.Series,
    horizon_days: int,
    atr: pd.Series | None = None,
) -> pd.Series:
    """
    Create the 3-class target variable.

    Classes:
      2 = Buy  (future return > buy_threshold)
      1 = Hold (in between)
      0 = Sell (future return < sell_threshold)

    If *atr* is provided, thresholds are scaled by normalised ATR
    so volatile stocks need bigger moves to trigger Buy/Sell.
    """
    future_return = close.shift(-horizon_days) / close - 1.0

    buy_thresh  = DEFAULT_BUY_THRESHOLD
    sell_thresh = DEFAULT_SELL_THRESHOLD

    # ATR-adaptive thresholds
    if atr is not None and not atr.empty:
        atr_norm = atr / close
        median_atr = atr_norm.median()
        if median_atr > 0:
            scale = atr_norm / median_atr
            scale = scale.clip(0.5, 3.0)
            buy_thresh_series  = buy_thresh * scale
            sell_thresh_series = sell_thresh * scale
            target = pd.Series(1, index=close.index, dtype=int)  # Default Hold
            target[future_return > buy_thresh_series] = 2
            target[future_return < sell_thresh_series] = 0
            target[future_return.isna()] = np.nan
            return target

    target = pd.Series(1, index=close.index, dtype=int)
    target[future_return > buy_thresh] = 2
    target[future_return < sell_thresh] = 0
    target[future_return.isna()] = np.nan
    return target


def get_predicted_price(current_price: float, predicted_return: float) -> float:
    """Convert predicted return to predicted price."""
    return round(current_price * (1 + predicted_return), 2)
