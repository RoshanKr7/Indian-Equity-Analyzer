"""
Ensemble predictor — combines all models into final predictions.

Full Model Lineup (August 2025 GPU upgrade):
  xgboost          → XGBoost 3.2, local, CPU-native
  lightgbm         → LightGBM DART, local, CPU-native
  nhits            → N-HiTS (Kaggle-trained), replaces Prophet, 50x faster
  prophet          → Fallback ONLY if nhits weights absent (0% weight normally)
  xgboost_temporal → Always-on sliding-window sequential (local, no weights needed)
  tft              → Temporal Fusion Transformer (Kaggle-trained), replaces LSTM slot
  lstm             → BiLSTM+Attention fallback if tft absent (0% weight when tft present)
  chronos          → Amazon Chronos-Bolt zero-shot
  sentiment        → FinancialBERT-Indian (fine-tuned) / FinancialBERT-2023 / FinBERT
  fundamental      → Factor-scored fundamentals

Fallback Logic (automatic — no user action needed):
  - tft absent    → its weight flows to xgboost_temporal (then lstm if bilstm exists)
  - nhits absent  → its weight flows to prophet fallback
  - No weights at all → xgboost + lightgbm + xgboost_temporal + chronos run fine

AI Toggle:
  When ai_enabled=False, returns technical+fundamental signals only.
"""

import numpy as np
import pandas as pd

from config.settings import (
    TIMEFRAMES, ENSEMBLE_WEIGHTS,
    MIN_CONFIDENCE_THRESHOLD,
    AGREEMENT_BONUS, DISAGREEMENT_CAP, LOW_DATA_PENALTY,
)
from models.catboost_model    import XGBoostPredictor
from models.lightgbm_model    import LightGBMPredictor
from models.prophet_model     import ProphetPredictor
from models.lstm_model        import LSTMPredictor
from models.xgboost_temporal_model import XGBoostTemporalPredictor
from models.chronos_model     import ChronosPredictor
from models.tft_model         import TFTPredictor
from models.nhits_model       import NHiTSPredictor
from models.feature_engineer  import build_features, build_target, get_predicted_price


class EnsemblePredictor:
    """Orchestrates all models and produces calibrated predictions."""

    def __init__(self):
        self.xgboost          = XGBoostPredictor()
        self.lightgbm         = LightGBMPredictor()
        self.prophet          = ProphetPredictor()        # fallback for nhits
        self.lstm             = LSTMPredictor()           # fallback for tft
        self.xgboost_temporal = XGBoostTemporalPredictor()
        self.chronos          = ChronosPredictor()
        self.tft              = TFTPredictor()
        self.nhits            = NHiTSPredictor()

    def predict_all_timeframes(
        self,
        stock_df:          pd.DataFrame,
        market_features:   pd.DataFrame | None,
        fundamental_score: float,
        sentiment_result:  dict,
        current_price:     float,
        ai_enabled:        bool = True,
    ) -> dict:
        """
        Run all models for every timeframe and combine with calibrated weighting.

        Parameters
        ----------
        ai_enabled : bool
            If False, skips all ML predictions — returns technical+fundamental signals.
        """
        if not ai_enabled:
            return self._technical_only_results(
                stock_df, current_price, sentiment_result, fundamental_score
            )

        features = build_features(stock_df, market_features)
        close    = stock_df["Close"]
        atr      = stock_df.get("ATR_14")
        has_enough_data = len(stock_df) >= 504  # ~2 years

        # Detect which GPU-trained models have weights
        tft_available   = self.tft.check_weights_exist()
        nhits_available = self.nhits.check_weights_exist()
        lstm_available  = self.lstm.check_weights_exist()

        results = {}

        for tf_code, tf_info in TIMEFRAMES.items():
            horizon = tf_info["days"]
            group   = tf_info["group"]
            weights = dict(ENSEMBLE_WEIGHTS[group])   # mutable copy

            target = build_target(close, horizon, atr)

            components = {}

            # ── Always-on models (no weights needed) ──────────────────────
            components["xgboost"]          = self.xgboost.predict(features, target, horizon)
            components["lightgbm"]         = self.lightgbm.predict(features, target, horizon)
            components["xgboost_temporal"] = self.xgboost_temporal.predict(features, target, horizon)
            components["chronos"]          = self.chronos.predict(close, horizon)

            # ── TFT (Kaggle-trained sequential) ────────────────────────────
            if tft_available:
                tft_result = self.tft.predict(features, horizon)
                if tft_result:
                    components["tft"] = tft_result
                else:
                    tft_available = False

            if not tft_available:
                # Redistribute TFT weight → xgboost_temporal (and lstm if available)
                tft_w = weights.get("tft", 0)
                if lstm_available:
                    lstm_result = self.lstm.predict(features, horizon)
                    if lstm_result:
                        components["lstm"] = lstm_result
                        # Split TFT weight 60/40 between lstm and xgboost_temporal
                        weights["lstm"]             = weights.get("lstm", 0) + tft_w * 0.6
                        weights["xgboost_temporal"] = weights.get("xgboost_temporal", 0) + tft_w * 0.4
                    else:
                        weights["xgboost_temporal"] = weights.get("xgboost_temporal", 0) + tft_w
                else:
                    weights["xgboost_temporal"] = weights.get("xgboost_temporal", 0) + tft_w
                weights["tft"] = 0.0

            # ── N-HiTS (Kaggle-trained trend) ─────────────────────────────
            nhits_result = self.nhits.predict(features, close, horizon)
            if nhits_available and nhits_result.get("confidence", 0) > 0.25:
                components["nhits"] = nhits_result
            else:
                # Fallback: use Prophet, redirect nhits weight to prophet
                prophet_result = self.prophet.predict(close, horizon)
                components["prophet"] = prophet_result
                nhits_w = weights.get("nhits", 0)
                weights["prophet"] = weights.get("prophet", 0) + nhits_w
                weights["nhits"]   = 0.0

            # ── Sentiment ──────────────────────────────────────────────────
            sent_score = sentiment_result.get("score", 0)
            sent_signal = "Buy" if sent_score > 0.15 else "Sell" if sent_score < -0.15 else "Hold"
            sent_conf   = min(abs(sent_score) * 2, 0.9)
            components["sentiment"] = {
                "signal":           sent_signal,
                "confidence":       round(sent_conf, 4),
                "predicted_return": round(sent_score * 0.03 * (horizon / 30), 4),
                "model":            sentiment_result.get("model", "FinancialBERT"),
            }

            # ── Fundamental ────────────────────────────────────────────────
            fund_signal = "Buy" if fundamental_score > 0.6 else "Sell" if fundamental_score < 0.4 else "Hold"
            fund_conf   = abs(fundamental_score - 0.5) * 2
            components["fundamental"] = {
                "signal":           fund_signal,
                "confidence":       round(fund_conf, 4),
                "predicted_return": round((fundamental_score - 0.5) * 0.1 * (horizon / 30), 4),
            }

            # ── Weighted ensemble ──────────────────────────────────────────
            signal_scores = {"Buy": 0.0, "Hold": 0.0, "Sell": 0.0}
            total_conf    = 0.0
            total_return  = 0.0

            for model_name, w in weights.items():
                if w == 0.0:
                    continue
                comp = components.get(model_name, {})
                sig  = comp.get("signal", "Hold")
                conf = comp.get("confidence", 0.33)
                ret  = comp.get("predicted_return", 0.0)
                signal_scores[sig] += w * conf
                total_conf         += w * conf
                total_return       += w * ret

            final_signal     = max(signal_scores, key=signal_scores.get)
            final_confidence = total_conf

            # ── Calibration ────────────────────────────────────────────────
            active_signals = [
                c.get("signal", "Hold")
                for c in components.values()
                if c.get("confidence", 0) > 0.35
            ]
            if active_signals:
                most_common = max(set(active_signals), key=active_signals.count)
                agreement   = active_signals.count(most_common) / len(active_signals)
                if agreement >= 0.67:
                    final_confidence += AGREEMENT_BONUS
                elif agreement <= 0.4:
                    final_confidence = min(final_confidence, DISAGREEMENT_CAP)

            if not has_enough_data:
                final_confidence -= LOW_DATA_PENALTY

            final_confidence = max(0.05, min(0.95, final_confidence))

            # Predicted price — blend direct price predictions
            predicted_price = get_predicted_price(current_price, total_return)
            price_sources   = []

            nhits_comp    = components.get("nhits", {})
            prophet_comp  = components.get("prophet", {})
            chronos_comp  = components.get("chronos", {})

            if nhits_comp.get("predicted_price"):
                price_sources.append((nhits_comp["predicted_price"], 0.40))
            elif prophet_comp.get("predicted_price"):
                price_sources.append((prophet_comp["predicted_price"], 0.40))
            if chronos_comp.get("predicted_price"):
                price_sources.append((chronos_comp["predicted_price"], 0.25))

            if price_sources:
                remaining_weight = 1.0 - sum(w for _, w in price_sources)
                blend = predicted_price * remaining_weight
                for price, w in price_sources:
                    blend += price * w
                predicted_price = round(blend, 2)

            results[tf_code] = {
                "signal":           final_signal,
                "confidence":       round(final_confidence, 4),
                "predicted_price":  predicted_price,
                "predicted_return": round(total_return, 4),
                "current_price":    round(current_price, 2),
                "timeframe":        tf_info["label"],
                "component_results": components,
                "gated":            final_confidence < MIN_CONFIDENCE_THRESHOLD,
                "ai_enabled":       True,
                "tft_available":    tft_available,
                "nhits_available":  nhits_available,
                "lstm_available":   lstm_available,
            }

        return results

    # ── AI-Off: Technical + Fundamentals only ─────────────────────────────
    def _technical_only_results(
        self,
        stock_df:          pd.DataFrame,
        current_price:     float,
        sentiment_result:  dict,
        fundamental_score: float,
    ) -> dict:
        """Used when AI toggle is OFF. No ML models run."""
        from core.technical_analysis import compute_technical_score
        tech_score = compute_technical_score(stock_df) if hasattr(stock_df, "columns") else 0.5
        composite  = tech_score * 0.6 + fundamental_score * 0.4
        signal     = "Buy" if composite > 0.6 else "Sell" if composite < 0.4 else "Hold"
        confidence = max(0.2, min(0.6, abs(composite - 0.5) * 1.5))

        results = {}
        for tf_code, tf_info in TIMEFRAMES.items():
            horizon         = tf_info["days"]
            predicted_return = (composite - 0.5) * 0.05 * (horizon / 30)
            results[tf_code] = {
                "signal":           signal,
                "confidence":       round(confidence, 4),
                "predicted_price":  get_predicted_price(current_price, predicted_return),
                "predicted_return": round(predicted_return, 4),
                "current_price":    round(current_price, 2),
                "timeframe":        tf_info["label"],
                "component_results": {
                    "technical":   {"signal": signal, "confidence": round(tech_score, 4),
                                    "predicted_return": round(predicted_return * 0.6, 4)},
                    "fundamental": {"signal": fund_signal if False else
                                    "Buy" if fundamental_score > 0.6 else
                                    "Sell" if fundamental_score < 0.4 else "Hold",
                                    "confidence": round(abs(fundamental_score - 0.5) * 2, 4),
                                    "predicted_return": round(predicted_return * 0.4, 4)},
                },
                "gated":        False,
                "ai_enabled":   False,
                "tft_available":  False,
                "nhits_available": False,
                "lstm_available":  False,
            }
        return results
