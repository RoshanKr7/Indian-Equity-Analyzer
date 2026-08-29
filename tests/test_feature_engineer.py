"""Tests for feature engineering pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd


def test_build_features_shape():
    """Feature matrix should have the expected columns."""
    from models.feature_engineer import build_features

    # Create dummy OHLCV with technical indicator columns
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)

    df = pd.DataFrame({
        "Open": prices + np.random.randn(300) * 0.2,
        "High": prices + abs(np.random.randn(300)),
        "Low": prices - abs(np.random.randn(300)),
        "Close": prices,
        "Volume": np.random.randint(1000, 100000, 300),
    }, index=dates)

    # Add some indicator columns that build_features expects
    df["RSI_14"] = 50 + np.random.randn(300) * 10
    df["MACD"] = np.random.randn(300)
    df["MACD_Signal"] = np.random.randn(300)
    df["MACD_Hist"] = np.random.randn(300)
    df["BB_PctB"] = np.random.rand(300)
    df["ATR_14"] = abs(np.random.randn(300)) + 1
    df["ADX_14"] = 20 + np.random.rand(300) * 30
    df["Williams_R"] = -50 + np.random.randn(300) * 20
    df["CCI_20"] = np.random.randn(300) * 100
    df["Volume_Ratio"] = 0.8 + np.random.rand(300) * 0.4
    df["Volume_Trend"] = 0.9 + np.random.rand(300) * 0.2
    df["OBV"] = np.cumsum(np.random.randn(300) * 1000)
    df["StochRSI"] = np.random.rand(300)
    df["Price_vs_SMA50"] = np.random.randn(300) * 5
    df["Price_vs_SMA200"] = np.random.randn(300) * 10
    df["Drawdown_52w"] = -abs(np.random.randn(300)) * 5

    features = build_features(df)

    assert len(features) == len(df)
    assert features.shape[1] >= 20  # Should have 20+ features
    assert not features.isnull().any().any()  # No NaN after cleaning


def test_build_target_classes():
    """Target should produce 3 classes."""
    from models.feature_engineer import build_target

    close = pd.Series(
        [100, 110, 90, 105, 95, 108, 92, 103, 97, 115],
        index=pd.date_range("2020-01-01", periods=10, freq="B"),
    )
    target = build_target(close, horizon_days=2)

    # Should have 3 classes (0, 1, 2) and NaN at the end
    valid = target.dropna()
    assert set(valid.unique()).issubset({0, 1, 2})


if __name__ == "__main__":
    test_build_features_shape()
    print("✅ test_build_features_shape passed")
    test_build_target_classes()
    print("✅ test_build_target_classes passed")
    print("All feature engineering tests passed!")
