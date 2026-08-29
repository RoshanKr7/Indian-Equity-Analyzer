"""
Historical OHLCV and fundamental data fetcher.

Uses yfinance with Streamlit caching so repeated requests within the
same session are instant.  TTL = 1 hour keeps data reasonably fresh
without hammering Yahoo Finance.
"""

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical(symbol: str, period: str = "10y") -> pd.DataFrame:
    """
    Download historical OHLCV data for *symbol*.

    Parameters
    ----------
    symbol : str
        Full Yahoo symbol (e.g. "RELIANCE.NS").
    period : str
        yfinance period string ("1y", "5y", "10y", "max").

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume.
        Index: DatetimeIndex (timezone-naive, date only).
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, auto_adjust=True)

    if df.empty:
        return pd.DataFrame()

    # Clean up: drop unnecessary columns, ensure numeric types
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"

    # Drop rows with NaN close
    df.dropna(subset=["Close"], inplace=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    """
    Extract key fundamental metrics from yfinance.

    Returns a flat dict with human-friendly keys.
    Missing values are returned as None (never raises).
    """
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        info = {}

    def _get(key, fmt=None):
        val = info.get(key)
        if val is None:
            return None
        if fmt == "pct":
            return round(val * 100, 2) if isinstance(val, (int, float)) else None
        if fmt == "round2":
            return round(val, 2) if isinstance(val, (int, float)) else None
        return val

    return {
        # -- Valuation --
        "pe_trailing":      _get("trailingPE", "round2"),
        "pe_forward":       _get("forwardPE", "round2"),
        "pb_ratio":         _get("priceToBook", "round2"),
        "peg_ratio":        _get("pegRatio", "round2"),
        "market_cap":       _get("marketCap"),
        "enterprise_value": _get("enterpriseValue"),

        # -- Profitability --
        "roe":              _get("returnOnEquity", "pct"),
        "roa":              _get("returnOnAssets", "pct"),
        "profit_margin":    _get("profitMargins", "pct"),
        "operating_margin": _get("operatingMargins", "pct"),
        "eps_trailing":     _get("trailingEps", "round2"),
        "eps_forward":      _get("forwardEps", "round2"),

        # -- Financial Health --
        "debt_to_equity":   _get("debtToEquity", "round2"),
        "current_ratio":    _get("currentRatio", "round2"),
        "quick_ratio":      _get("quickRatio", "round2"),

        # -- Growth --
        "revenue_growth":   _get("revenueGrowth", "pct"),
        "earnings_growth":  _get("earningsGrowth", "pct"),

        # -- Dividend --
        "dividend_yield":   _get("dividendYield", "pct"),
        "payout_ratio":     _get("payoutRatio", "pct"),

        # -- Other --
        "beta":             _get("beta", "round2"),
        "52w_high":         _get("fiftyTwoWeekHigh", "round2"),
        "52w_low":          _get("fiftyTwoWeekLow", "round2"),
        "50d_avg":          _get("fiftyDayAverage", "round2"),
        "200d_avg":         _get("twoHundredDayAverage", "round2"),
        "avg_volume":       _get("averageVolume"),
        "sector":           _get("sector"),
        "industry":         _get("industry"),
        "description":      _get("longBusinessSummary"),
    }
