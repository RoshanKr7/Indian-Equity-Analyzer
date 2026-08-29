"""
Market context data — Nifty 50, India VIX, sectoral indices.

These macro features give our models information about the *environment*
the stock trades in, not just the stock itself.  Research shows that
market regime (bull / bear / sideways) and volatility state are among
the strongest predictors of individual stock returns at every horizon.
"""

import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np

from config.settings import MARKET_TICKERS, SECTOR_INDEX_MAP, DEFAULT_SECTOR_INDEX


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(period: str = "10y") -> dict[str, pd.DataFrame]:
    """
    Download Nifty 50, India VIX, and sector index histories.

    Returns
    -------
    dict
        Mapping of identifier → OHLCV DataFrame.
        e.g. {"nifty50": df, "india_vix": df, ...}
    """
    result = {}
    for name, symbol in MARKET_TICKERS.items():
        try:
            df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df.index.name = "Date"
                result[name] = df[["Close"]].rename(columns={"Close": name})
        except Exception:
            pass
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_index(sector: str, period: str = "10y") -> pd.Series | None:
    """Fetch the sector-specific index for relative-strength calculation."""
    symbol = SECTOR_INDEX_MAP.get(sector, DEFAULT_SECTOR_INDEX)
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df["Close"].rename("sector_index")
    except Exception:
        return None


def compute_market_features(
    stock_df: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    sector: str | None = None,
    sector_index: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build market-context features aligned to the stock's date index.

    Features produced:
      - nifty50_ret_1d / 5d / 20d     : market return at different windows
      - india_vix                       : absolute VIX level
      - india_vix_chg_5d / 20d         : VIX rate-of-change
      - sector_rel_strength_20d        : stock return minus sector return (20d)
      - day_sin, day_cos               : day-of-week cyclical encoding
      - month_sin, month_cos           : month-of-year cyclical encoding

    All features are forward-filled then back-filled so there are no NaNs
    at the cost of a slight information mismatch at the very start.
    """
    features = pd.DataFrame(index=stock_df.index)

    # --- Nifty 50 returns ---
    if "nifty50" in market_data:
        nifty = market_data["nifty50"].reindex(stock_df.index, method="ffill")
        for window in [1, 5, 20]:
            features[f"nifty50_ret_{window}d"] = nifty["nifty50"].pct_change(window)

    # --- India VIX ---
    if "india_vix" in market_data:
        vix = market_data["india_vix"].reindex(stock_df.index, method="ffill")
        features["india_vix"] = vix["india_vix"]
        features["india_vix_chg_5d"]  = vix["india_vix"].pct_change(5)
        features["india_vix_chg_20d"] = vix["india_vix"].pct_change(20)

    # --- Sector relative strength ---
    if sector_index is not None:
        sec = sector_index.reindex(stock_df.index, method="ffill")
        stock_ret_20 = stock_df["Close"].pct_change(20)
        sec_ret_20   = sec.pct_change(20)
        features["sector_rel_strength_20d"] = stock_ret_20 - sec_ret_20

    # --- Cyclical time features ---
    dow = stock_df.index.dayofweek  # 0=Monday
    features["day_sin"]   = np.sin(2 * np.pi * dow / 5)
    features["day_cos"]   = np.cos(2 * np.pi * dow / 5)
    month = stock_df.index.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)

    features.ffill(inplace=True)
    features.bfill(inplace=True)
    return features
