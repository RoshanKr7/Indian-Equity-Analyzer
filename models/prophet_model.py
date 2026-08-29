"""
Prophet time-series forecaster.

Why Prophet?
  - Built by Meta for business time-series with strong seasonality.
  - Handles missing data (market holidays) gracefully.
  - Native support for custom holidays (Indian market holidays).
  - Provides uncertainty intervals → we derive confidence from interval width.
  - Best for medium-term (1-6 month) trend forecasting.
"""

import pandas as pd
import numpy as np
from prophet import Prophet

from config.settings import (
    PROPHET_YEARLY_SEASONALITY,
    PROPHET_WEEKLY_SEASONALITY,
    PROPHET_DAILY_SEASONALITY,
)

# Indian market holidays (major ones)
INDIAN_HOLIDAYS = pd.DataFrame({
    "holiday": "indian_market",
    "ds": pd.to_datetime([
        # Republic Day, Holi, Good Friday, Eid, Independence Day,
        # Ganesh Chaturthi, Dussehra, Diwali, Christmas
        # We add a few years of dates; Prophet handles date matching
        "2024-01-26", "2024-03-25", "2024-03-29", "2024-04-11",
        "2024-08-15", "2024-09-07", "2024-10-12", "2024-11-01", "2024-12-25",
        "2025-01-26", "2025-03-14", "2025-04-18", "2025-03-31",
        "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-20", "2025-12-25",
        "2026-01-26", "2026-03-03", "2026-04-03", "2026-03-20",
        "2026-08-15", "2026-09-16", "2026-10-02", "2026-11-08", "2026-12-25",
    ]),
    "lower_window": 0,
    "upper_window": 0,
})


class ProphetPredictor:
    """Facebook Prophet forecaster for stock price trends."""

    def __init__(self):
        self.name = "prophet"

    def predict(self, close_series: pd.Series, horizon_days: int) -> dict:
        """
        Fit Prophet on close prices and forecast *horizon_days* ahead.

        Returns signal, confidence, predicted return and price.
        """
        if len(close_series) < 252:  # Need at least 1 year
            return self._default_result(close_series)

        # Prophet requires columns: ds, y
        prophet_df = pd.DataFrame({
            "ds": close_series.index,
            "y": close_series.values,
        })

        model = Prophet(
            yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
            weekly_seasonality=PROPHET_WEEKLY_SEASONALITY,
            daily_seasonality=PROPHET_DAILY_SEASONALITY,
            holidays=INDIAN_HOLIDAYS,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10,
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)

        # Get the forecasted value at the horizon
        last_actual = close_series.iloc[-1]
        horizon_row = forecast.iloc[-1]
        predicted_price = horizon_row["yhat"]
        upper = horizon_row["yhat_upper"]
        lower = horizon_row["yhat_lower"]

        predicted_return = (predicted_price - last_actual) / last_actual

        # Confidence from uncertainty interval width
        interval_width = (upper - lower) / last_actual
        confidence = max(0.2, 1.0 - interval_width)
        confidence = min(confidence, 0.95)

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
            "predicted_price": round(predicted_price, 2),
            "upper_bound": round(upper, 2),
            "lower_bound": round(lower, 2),
        }

    def _default_result(self, close_series):
        price = close_series.iloc[-1] if len(close_series) > 0 else 0
        return {
            "signal": "Hold",
            "confidence": 0.3,
            "predicted_return": 0.0,
            "predicted_price": round(price, 2),
            "upper_bound": round(price * 1.05, 2),
            "lower_bound": round(price * 0.95, 2),
        }
