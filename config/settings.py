"""
Centralised configuration for the Indian Equity Prediction Tool.

Every tunable knob lives here so the rest of the codebase can import
`from config.settings import ...` instead of scattering magic numbers.
"""

# ──────────────────────────────────────────────
# Prediction Timeframes
# ──────────────────────────────────────────────
TIMEFRAMES = {
    "7d":  {"days": 7,    "group": "short",  "label": "7 Days"},
    "15d": {"days": 15,   "group": "short",  "label": "15 Days"},
    "1m":  {"days": 30,   "group": "medium", "label": "1 Month"},
    "3m":  {"days": 90,   "group": "medium", "label": "3 Months"},
    "6m":  {"days": 180,  "group": "long",   "label": "6 Months"},
    "1y":  {"days": 365,  "group": "long",   "label": "1 Year"},
    "3y":  {"days": 1095, "group": "long",   "label": "3 Years"},
}

# ──────────────────────────────────────────────
# Ensemble Weights — per timeframe group
# ──────────────────────────────────────────────
# Keys must match the model slugs returned by each predictor.
# Model lineup (August 2025 GPU upgrade):
#   xgboost          → replaces catboost (XGBoost 3.2, already installed)
#   lightgbm         → DART boosting (upgraded)
#   nhits            → replaces prophet (50x faster, better multi-horizon)
#   prophet          → kept as fallback ONLY if nhits weights absent (0% weight)
#   xgboost_temporal → local always-on sequential fallback
#   tft              → replaces lstm slot (Temporal Fusion Transformer, GPU-trained)
#   lstm             → BiLSTM fallback if tft absent (0% weight when tft present)
#   chronos          → zero-shot independent opinion
#   sentiment        → FinancialBERT-Indian (fine-tuned) or FinancialBERT-2023
#   fundamental      → factor-scored fundamentals
#
# Weight redistribution at runtime:
#   - If tft absent: tft weight → xgboost_temporal (+ lstm if bilstm weights exist)
#   - If nhits absent: nhits weight → prophet fallback
ENSEMBLE_WEIGHTS = {
    "short": {
        "xgboost":           0.28,
        "lightgbm":          0.18,
        "nhits":             0.08,  # N-HiTS (50x faster than prophet, better)
        "prophet":           0.00,  # Fallback only
        "xgboost_temporal":  0.12,  # Always-on sequential
        "tft":               0.17,  # TFT (when Kaggle weights present)
        "lstm":              0.00,  # Fallback if tft absent
        "chronos":           0.08,
        "sentiment":         0.09,
        "fundamental":       0.00,
    },
    "medium": {
        "xgboost":           0.20,
        "lightgbm":          0.15,
        "nhits":             0.20,  # N-HiTS stronger at medium-term trends
        "prophet":           0.00,
        "xgboost_temporal":  0.08,
        "tft":               0.13,
        "lstm":              0.00,
        "chronos":           0.10,
        "sentiment":         0.07,
        "fundamental":       0.07,
    },
    "long": {
        "xgboost":           0.10,
        "lightgbm":          0.08,
        "nhits":             0.12,  # N-HiTS handles long-horizon well
        "prophet":           0.00,
        "xgboost_temporal":  0.05,
        "tft":               0.08,
        "lstm":              0.00,
        "chronos":           0.05,
        "sentiment":         0.04,
        "fundamental":       0.48,  # Fundamentals dominate long-term
    },
}

# ──────────────────────────────────────────────
# Confidence & Classification
# ──────────────────────────────────────────────
# If ensemble confidence < this threshold the timeframe is hidden.
MIN_CONFIDENCE_THRESHOLD = 0.30

# Default return thresholds for Buy/Hold/Sell classification.
# At runtime these are adjusted by the stock's ATR to account for
# different volatility regimes.
DEFAULT_BUY_THRESHOLD  =  0.05   #  +5 %
DEFAULT_SELL_THRESHOLD = -0.05   #  -5 %

# Agreement / disagreement calibration
AGREEMENT_BONUS   = 0.10   # +10 % confidence if ≥4 models agree
DISAGREEMENT_CAP  = 0.40   # cap confidence at 40 % if 50/50 split
LOW_DATA_PENALTY  = 0.20   # –20 % confidence if <2 yr of history

# ──────────────────────────────────────────────
# AI Toggle
# ──────────────────────────────────────────────
AI_ENABLED_DEFAULT = True   # Default: AI predictions are ON

# ──────────────────────────────────────────────
# Market Context Tickers (yfinance symbols)
# ──────────────────────────────────────────────
MARKET_TICKERS = {
    "nifty50":    "^NSEI",
    "india_vix":  "^INDIAVIX",
    "nifty_bank": "^NSEBANK",
    "nifty_it":   "^CNXIT",
}

# Map stock sectors (from yfinance) to the sectoral index ticker
# so we can compute sector-relative strength.
SECTOR_INDEX_MAP = {
    "Financial Services": "^NSEBANK",
    "Information Technology": "^CNXIT",
    # For sectors without a clean yfinance index we fall back to Nifty 50
}
DEFAULT_SECTOR_INDEX = "^NSEI"

# ──────────────────────────────────────────────
# Technical Indicators to compute
# ──────────────────────────────────────────────
TECHNICAL_INDICATORS = [
    "RSI_14", "StochRSI",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_PctB",
    "SMA_20", "SMA_50", "SMA_200",
    "EMA_12", "EMA_26",
    "ATR_14",
    "OBV",
    "ADX_14",
    "Williams_R",
    "CCI_20",
]

# ──────────────────────────────────────────────
# Fundamental Scoring — factor weights
# ──────────────────────────────────────────────
FUNDAMENTAL_WEIGHTS = {
    "valuation":       0.25,
    "profitability":   0.25,
    "financial_health": 0.20,
    "growth":          0.20,
    "dividend":        0.10,
}

# ──────────────────────────────────────────────
# Sentiment / News
# ──────────────────────────────────────────────
# Priority 1: Fine-tuned FinancialBERT on Indian NSE/BSE news (Kaggle fine-tuned)
# Priority 2: FinancialBERT-2023 (generic English financial news)
# Priority 3: FinBERT-2019 (original fallback)
FINBERT_INDIAN_PATH      = "models/pretrained/finbert_indian"  # local fine-tuned dir
FINBERT_MODEL            = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
FINBERT_MODEL_FALLBACK   = "ProsusAI/finbert"
NEWS_HEADLINE_COUNT      = 15
GOOGLE_NEWS_RSS          = "https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"

# Optional: Qwen2.5-1.5B-Instruct for enhanced sentiment reasoning
# Only loads if user enables "Enhanced AI Sentiment" in sidebar
# Model size: ~3GB. Runs on CPU (slow but works). Use 4-bit quant for speed.
QWEN_SENTIMENT_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_SENTIMENT_ENABLED   = False  # Default: off (user can enable in sidebar)

# ──────────────────────────────────────────────
# BiLSTM + Attention Config
# (fallback when TFT weights absent)
# ──────────────────────────────────────────────
LSTM_HIDDEN_SIZE  = 128
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.3
LSTM_SEQUENCE_LEN = 60   # 60-day lookback window
LSTM_WEIGHTS_DIR  = "models/pretrained"

# ──────────────────────────────────────────────
# TFT Config (Temporal Fusion Transformer)
# ──────────────────────────────────────────────
TFT_WEIGHTS_DIR   = "models/pretrained"   # tft_weights.pth + tft_config.json
TFT_D_MODEL       = 64                    # Embedding dimension (matches Kaggle notebook)
TFT_N_HEADS       = 4
TFT_N_LSTM_LAYERS = 2
TFT_DROPOUT       = 0.2
TFT_SEQUENCE_LEN  = 60                   # Same as BiLSTM for consistency

# ──────────────────────────────────────────────
# N-HiTS Config (replaces Prophet)
# ──────────────────────────────────────────────
NHITS_WEIGHTS_DIR  = "models/pretrained"  # nhits_weights.pth + nhits_config.json
NHITS_CONTEXT_LEN  = 120                  # 120 trading days (~6 months) of history
NHITS_MAX_HORIZON  = 365                  # Max days ahead to predict (covers all timeframes)
NHITS_D_HIDDEN     = 256
NHITS_N_LAYERS     = 2
NHITS_POOLING_SIZES = [5, 2, 1]          # Coarse, medium, fine blocks

# ──────────────────────────────────────────────
# XGBoost Temporal Config (local always-on fallback)
# ──────────────────────────────────────────────
XGBT_SEQUENCE_LEN  = 20   # 20-day window (shorter to keep feature space manageable)
XGBT_N_ESTIMATORS  = 300  # Fewer trees — wide feature space, faster training

# ──────────────────────────────────────────────
# Chronos Config
# ──────────────────────────────────────────────
CHRONOS_MODEL_ID = "amazon/chronos-bolt-small"  # ~40 M params, fast
CHRONOS_CONTEXT_LENGTH = 512

# ──────────────────────────────────────────────
# Prophet Config
# ──────────────────────────────────────────────
PROPHET_YEARLY_SEASONALITY  = True
PROPHET_WEEKLY_SEASONALITY  = True
PROPHET_DAILY_SEASONALITY   = False

# ──────────────────────────────────────────────
# Walk-Forward Validation
# ──────────────────────────────────────────────
WF_MIN_TRAIN_DAYS  = 504   # ~2 years of trading days
WF_N_SPLITS        = 5
WF_PURGE_DAYS      = 5     # Extra gap to prevent label leakage

# ──────────────────────────────────────────────
# UI / Report
# ──────────────────────────────────────────────
APP_TITLE     = "🇮🇳 Indian Equity Analyzer"
APP_ICON      = "📈"
REPORT_TITLE  = "Indian Equity Analysis Report"
DISCLAIMER    = (
    "⚠️ DISCLAIMER: This tool is for educational and informational purposes only. "
    "It does NOT constitute financial advice. Stock markets are inherently risky and "
    "past performance does not guarantee future results. Always consult a qualified "
    "financial advisor before making investment decisions. The creators of this tool "
    "are not liable for any financial losses incurred."
)
