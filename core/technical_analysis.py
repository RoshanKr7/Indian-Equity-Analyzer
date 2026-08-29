"""
Technical indicator computation and aggregate scoring.

Uses the ``ta`` library for reliable indicator calculations.
"""

import pandas as pd
import numpy as np
import ta


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 15+ technical indicator columns to a copy of *df*."""
    out = df.copy()
    c, h, l, v = out["Close"], out["High"], out["Low"], out["Volume"]

    # Trend
    out["SMA_20"]  = ta.trend.sma_indicator(c, window=20)
    out["SMA_50"]  = ta.trend.sma_indicator(c, window=50)
    out["SMA_200"] = ta.trend.sma_indicator(c, window=200)
    out["EMA_12"]  = ta.trend.ema_indicator(c, window=12)
    out["EMA_26"]  = ta.trend.ema_indicator(c, window=26)

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    out["MACD"]        = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    out["MACD_Hist"]   = macd.macd_diff()
    adx_ind = ta.trend.ADXIndicator(h, l, c, window=14)
    out["ADX_14"] = adx_ind.adx()
    cci_ind = ta.trend.CCIIndicator(h, l, c, window=20)
    out["CCI_20"] = cci_ind.cci()

    # Momentum
    out["RSI_14"] = ta.momentum.rsi(c, window=14)
    stoch_rsi = ta.momentum.StochRSIIndicator(c, window=14, smooth1=3, smooth2=3)
    out["StochRSI"] = stoch_rsi.stochrsi()
    out["Williams_R"] = ta.momentum.williams_r(h, l, c, lbp=14)

    # Volatility
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    out["BB_Upper"] = bb.bollinger_hband()
    out["BB_Lower"] = bb.bollinger_lband()
    out["BB_PctB"]  = bb.bollinger_pband()
    atr_ind = ta.volatility.AverageTrueRange(h, l, c, window=14)
    out["ATR_14"] = atr_ind.average_true_range()

    # Volume
    out["OBV"] = ta.volume.on_balance_volume(c, v)
    out["Volume_Ratio"] = v / v.rolling(20).mean()
    out["Volume_Trend"] = v.rolling(5).mean() / v.rolling(20).mean()

    # Derived
    out["Price_vs_SMA50"]  = (c - out["SMA_50"]) / out["SMA_50"] * 100
    out["Price_vs_SMA200"] = (c - out["SMA_200"]) / out["SMA_200"] * 100
    rolling_high = c.rolling(252).max()
    out["Drawdown_52w"] = (c - rolling_high) / rolling_high * 100

    return out


def compute_technical_score(df: pd.DataFrame) -> float:
    """Aggregate technical signal from latest row into 0-1 score."""
    if df.empty:
        return 0.5
    latest = df.iloc[-1]
    signals = []

    rsi = latest.get("RSI_14")
    if rsi is not None and not np.isnan(rsi):
        signals.append(1.0 - min(max((rsi - 30) / 40, 0), 1))

    macd_hist = latest.get("MACD_Hist")
    if macd_hist is not None and not np.isnan(macd_hist):
        signals.append(1.0 if macd_hist > 0 else 0.0)

    pv50 = latest.get("Price_vs_SMA50")
    if pv50 is not None and not np.isnan(pv50):
        signals.append(min(max((pv50 + 10) / 20, 0), 1))

    pv200 = latest.get("Price_vs_SMA200")
    if pv200 is not None and not np.isnan(pv200):
        signals.append(min(max((pv200 + 15) / 30, 0), 1))

    pctb = latest.get("BB_PctB")
    if pctb is not None and not np.isnan(pctb):
        signals.append(1.0 - min(max(pctb, 0), 1))

    wr = latest.get("Williams_R")
    if wr is not None and not np.isnan(wr):
        signals.append(1.0 - min(max((wr + 80) / 60, 0), 1))

    vr = latest.get("Volume_Ratio")
    if vr is not None and not np.isnan(vr):
        signals.append(min(vr / 2, 1))

    return round(sum(signals) / len(signals), 4) if signals else 0.5
