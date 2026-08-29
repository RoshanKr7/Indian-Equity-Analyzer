"""
Amazon Chronos-Bolt — zero-shot time-series foundation model.

Why Chronos?
  - Pre-trained on BILLIONS of time-series data points (weather, finance,
    energy, retail, etc.) → strong general pattern recognition.
  - Zero-shot: no training needed, just pip install and run.
  - Provides probabilistic forecasts (quantiles) → confidence intervals.
  - Acts as an "independent second opinion" from a model that has never
    seen this specific stock, reducing overfitting risk.
  - ~40M parameters (Bolt-Small), runs on CPU in seconds.

This is the single biggest "free accuracy" gain in the pipeline.
"""

import numpy as np
import pandas as pd
import torch
import streamlit as st

from config.settings import CHRONOS_MODEL_ID, CHRONOS_CONTEXT_LENGTH


@st.cache_resource(show_spinner=False)
def _load_chronos():
    """Load Chronos pipeline once and cache globally."""
    try:
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            CHRONOS_MODEL_ID,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        return pipeline
    except Exception:
        return None


class ChronosPredictor:
    """Zero-shot time-series forecasting via Chronos-Bolt."""

    def __init__(self):
        self.name = "chronos"

    def predict(self, close_series: pd.Series, horizon_days: int) -> dict:
        """
        Generate probabilistic forecast for *horizon_days* ahead.

        Returns signal, confidence, predicted return and price.
        """
        pipeline = _load_chronos()
        if pipeline is None:
            return self._default_result(close_series)

        # Use last CONTEXT_LENGTH days
        context = close_series.iloc[-CHRONOS_CONTEXT_LENGTH:].values.astype(np.float32)
        context_tensor = torch.tensor(context).unsqueeze(0)

        try:
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=horizon_days,
                num_samples=20,
                limit_prediction_length=False,
            )
        except Exception:
            return self._default_result(close_series)

        # forecast shape: (1, num_samples, horizon_days)
        samples = forecast[0].numpy()  # (num_samples, horizon_days)

        # Take the last day's predictions
        final_day = samples[:, -1]
        median_pred = float(np.median(final_day))
        q10 = float(np.percentile(final_day, 10))
        q90 = float(np.percentile(final_day, 90))

        last_price = float(close_series.iloc[-1])
        predicted_return = (median_pred - last_price) / last_price

        # Confidence from prediction interval width
        interval_width = (q90 - q10) / last_price
        confidence = max(0.2, 1.0 - interval_width * 2)
        confidence = min(confidence, 0.90)

        if predicted_return > 0.02:
            signal = "Buy"
        elif predicted_return < -0.02:
            signal = "Sell"
        else:
            signal = "Hold"

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "predicted_return": round(predicted_return, 4),
            "predicted_price": round(median_pred, 2),
            "upper_bound": round(q90, 2),
            "lower_bound": round(q10, 2),
        }

    def _default_result(self, close_series):
        price = float(close_series.iloc[-1]) if len(close_series) > 0 else 0
        return {
            "signal": "Hold",
            "confidence": 0.25,
            "predicted_return": 0.0,
            "predicted_price": round(price, 2),
            "upper_bound": round(price * 1.05, 2),
            "lower_bound": round(price * 0.95, 2),
        }
