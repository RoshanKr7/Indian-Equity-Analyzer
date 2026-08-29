# Architecture & Design Decisions — AI Indian Equity Predictor

This document explains **every architectural choice, model selection, and design
decision** in the project. If you're wondering "why did we do X instead of Y?"
— the answer is here.

---

## Table of Contents

1. [Why Streamlit?](#1-why-streamlit)
2. [Data Source: yfinance](#2-data-source-yfinance)
3. [Ticker Validation Strategy](#3-ticker-validation-strategy)
4. [Model Selection Deep-Dive](#4-model-selection-deep-dive)
5. [Feature Engineering Philosophy](#5-feature-engineering-philosophy)
6. [Ensemble Weighting Strategy](#6-ensemble-weighting-strategy)
7. [Validation Methodology](#7-validation-methodology)
8. [Confidence Calibration](#8-confidence-calibration)
9. [Fundamental Scoring System](#9-fundamental-scoring-system)
10. [Sentiment Analysis Pipeline](#10-sentiment-analysis-pipeline)
11. [UI/UX Design Choices](#11-uiux-design-choices)
12. [PDF Report Generation](#12-pdf-report-generation)
13. [Performance & Caching](#13-performance--caching)
14. [Limitations & Honest Disclaimers](#14-limitations--honest-disclaimers)

---

## 1. Why Streamlit?

**Chosen over:** Flask + React, Dash, Gradio, raw HTML/JS

**Reasons:**
- **Native Python ecosystem**: All our ML models (CatBoost, LightGBM, Prophet,
  PyTorch) are Python libraries. Streamlit runs them natively without needing a
  separate backend API layer.
- **Session state**: Multi-step flows (enter ticker → confirm → analyze → view)
  are naturally handled by `st.session_state`.
- **Built-in caching**: `@st.cache_data` and `@st.cache_resource` prevent
  redundant API calls and model loading — critical since yfinance has rate limits.
- **Interactive charts**: Plotly integration is first-class with
  `st.plotly_chart()`.
- **Rapid prototyping**: We skip writing HTML/CSS/JS boilerplate for standard
  widgets (buttons, inputs, tabs, spinners, progress bars).
- **Cloud-ready**: Streamlit Community Cloud offers free hosting with one-click
  deploy from GitHub.

**Trade-offs accepted:**
- Less control over layout than raw HTML/CSS (mitigated with custom CSS).
- Every interaction reruns the script (mitigated with caching).
- Not ideal for high-frequency, real-time dashboards (but our minimum horizon
  is 7 days, so EOD data is perfectly fine).

---

## 2. Data Source: yfinance

**Chosen over:** Alpha Vantage, EODHD, Quandl, NSE direct scraping

**Reasons:**
- **Free with no API key**: Zero setup friction. Alpha Vantage has free tier
  but requires registration and has 5 calls/minute limit.
- **Comprehensive**: Provides OHLCV history (up to ~20 years), fundamental data
  (P/E, ROE, market cap, etc.), dividends, and splits.
- **Indian stock support**: Append `.NS` (NSE) or `.BO` (BSE) to tickers.
- **Community-maintained**: Actively developed on GitHub with rapid bug fixes.

**Limitations we handle:**
- Not real-time (EOD data) — fine for our 7-day+ horizons.
- Some fundamental fields return `None` for small-caps — we default to "N/A".
- Rate limiting on bulk calls — we use `@st.cache_data(ttl=3600)`.
- Unofficial scraper of Yahoo Finance — could break if Yahoo changes their site.

**Why not NSE direct?**
- NSE blocks automated requests aggressively (CAPTCHAs, IP bans).
- No official free API exists.
- yfinance provides a clean, stable interface.

---

## 3. Ticker Validation Strategy

**NSE-first, BSE-fallback** because:
- ~90% of retail trading volume in India is on NSE.
- Most users think in NSE symbols (RELIANCE, TCS, INFY).
- BSE symbols sometimes differ (especially for older companies).

**Equity-only filter**: We check `quoteType == "EQUITY"` to reject:
- ETFs (like NIFTYBEES)
- Mutual funds
- Index tickers

This prevents confusing results where someone enters an ETF thinking it's a stock.

---

## 4. Model Selection Deep-Dive

### 4.1 CatBoost (Gradient Boosting)

**Chosen over:** XGBoost, Random Forest

**Why CatBoost specifically?**
- **Symmetric (oblivious) decision trees**: Each node at the same depth uses
  the same split condition. This acts as built-in regularisation, making it
  significantly more resistant to overfitting on noisy financial data.
- **Ordered boosting**: During training, CatBoost uses a permutation-based
  scheme that prevents target leakage between training examples. In finance,
  where sequential data creates subtle leakage, this is critical.
- **Native categorical support**: Stock sector and industry are categorical
  features. XGBoost requires manual one-hot/label encoding (which can introduce
  ordinal bias). CatBoost handles them internally with target statistics.
- **Better out-of-box**: In benchmarks, CatBoost with default hyperparameters
  frequently matches or beats a tuned XGBoost.

**What about XGBoost?**
XGBoost is a great model, but its level-wise tree growth is more prone to
overfitting on financial data unless carefully regularised. CatBoost's symmetric
trees provide this guard by design.

### 4.2 LightGBM (Gradient Boosting)

**Purpose:** Ensemble diversity — provides a "second opinion" from a different
tree-building strategy.

**Why stack two gradient boosters?**
- CatBoost uses **symmetric** trees (same split at each depth).
- LightGBM uses **leaf-wise** growth (grows the leaf with max delta loss).
- These two strategies explore different parts of the feature space.
- Averaging their predictions reduces model-specific bias (variance reduction).
- This is standard practice in:
  - Kaggle competitions (top solutions almost always stack multiple GBMs)
  - Professional quant shops (ensemble diversity is a core principle)

### 4.3 Facebook Prophet

**Chosen over:** ARIMA, SARIMA, Holt-Winters

**Why Prophet?**
- **Additive decomposition**: Separates trend + seasonality + holidays.
  Indian markets have strong seasonality (Diwali rally, budget season, Q4 FII
  selling, March-end portfolio rebalancing).
- **Holiday support**: We feed Indian market holidays directly. ARIMA has no
  concept of holidays.
- **Missing data handling**: Prophet handles gaps (weekends, holidays) natively.
  ARIMA requires careful imputation.
- **Uncertainty intervals**: Prophet's Bayesian framework provides prediction
  intervals, which we convert to confidence scores.
- **No stationarity requirement**: ARIMA requires differencing to achieve
  stationarity. Prophet works on raw price data.

**What about ARIMA?**
ARIMA is a strong baseline but requires manual order selection (p, d, q),
stationarity testing, and has no concept of seasonality without SARIMA extension.
Prophet automates all of this.

### 4.4 LSTM (Long Short-Term Memory)

**Why LSTM over GRU or vanilla RNN?**
- **Gating mechanism**: LSTM's forget gate, input gate, and output gate allow
  it to learn which past information to retain or discard. Stock markets have
  both short-term momentum and long-term mean reversion — LSTM can capture both.
- **Gradient stability**: Vanilla RNNs suffer from vanishing gradients on long
  sequences (60-day lookback). LSTM's cell state provides a gradient highway.
- **GRU vs LSTM**: GRU is faster (fewer parameters) but slightly less expressive.
  Since we train on Kaggle GPU, the extra cost is negligible.

**Why train on Kaggle instead of locally?**
- LSTM training requires hundreds of epochs over 10 years of data.
- On CPU: ~2-3 hours. On Kaggle T4 GPU: ~15-20 minutes.
- We save the weights (`.pth`) and load for inference only — which is fast on CPU.

**Architecture choices:**
- **2 layers** (not 1, not 4): 1 layer is too shallow for financial patterns.
  4+ layers overfit on the limited data we have (~2500 trading days).
- **128 hidden units**: Sweet spot between expressiveness and overfitting.
  256 units on <3000 samples → definite overfitting.
- **60-day sequence length**: ~3 months of trading data. Captures quarterly
  patterns and medium-term momentum.
- **Dropout 0.3**: Standard regularisation for financial LSTM. Higher dropout
  (0.5) kills too much signal.

### 4.5 Amazon Chronos-Bolt

**Chosen over:** Google TimesFM, N-BEATS, TFT (Temporal Fusion Transformer)

**Why Chronos?**
- **Zero-shot**: No training needed. Period. Just `pip install` and run.
- **Foundation model**: Pre-trained on billions of time-series from diverse
  domains (weather, finance, energy, retail). It has seen more time-series
  patterns than we could ever train on with a single stock.
- **Probabilistic forecasts**: Returns quantile predictions, not just point
  estimates. We derive confidence from the prediction interval width.
- **Independent opinion**: Since it was never trained on *this specific stock*,
  it provides an unbiased baseline. If our trained models and Chronos agree,
  that's a strong signal. If they disagree, it's a warning.

**Why not Google TimesFM?**
- TimesFM requires installation from source (`git clone` + editable install).
- Chronos has a clean `pip install chronos-forecasting`.
- Both are comparable in accuracy, but Chronos is easier to integrate.

**Why not TFT or N-BEATS?**
- Both require training from scratch on your data.
- TFT is ~100M parameters — too heavy for local training.
- N-BEATS is lighter but still requires a training pipeline.
- Chronos gives us similar quality for zero effort.

### 4.6 FinBERT (Sentiment)

**Chosen over:** VADER, TextBlob, GPT-based sentiment

**Why FinBERT?**
- **Domain-specific**: Fine-tuned on financial text. VADER/TextBlob are
  general-purpose and miss financial nuances (e.g., "The company cut its
  dividend" → VADER might see "cut" as neutral).
- **Free**: HuggingFace model, no API calls, runs locally.
- **Fast**: ~1 second for a batch of 15 headlines.
- **Proven**: ProsusAI/finbert is cited in hundreds of academic papers.

**Why not GPT?**
- GPT-4/Claude require paid API calls (not free).
- FinBERT is purpose-built for this exact task and runs offline.

---

## 5. Feature Engineering Philosophy

### Why 30+ features?

Financial markets are driven by multiple overlapping forces:
- **Momentum** (recent returns predict near-future returns)
- **Mean reversion** (overextended moves snap back)
- **Volatility regimes** (high VIX = different dynamics)
- **Market beta** (individual stocks correlate with the market)
- **Seasonality** (budget season, earnings season, Diwali)

Each feature family captures a different force. Gradient boosting models are
excellent at selecting relevant features and ignoring noise, so more features
(within reason) generally help rather than hurt.

### Why log returns instead of raw prices?

Log returns are:
- **Stationary**: Raw prices are non-stationary (trend upward over time).
  ML models work better on stationary data.
- **Additive**: log(P_t / P_0) = sum of daily log returns. This makes
  multi-period return calculation consistent.
- **Symmetric**: A +50% gain and -50% loss are symmetric in log space.

### Why ATR-normalised thresholds?

A 5% move in Reliance (low volatility blue-chip) is a big deal.
A 5% move in a small-cap stock might be a normal Tuesday.

By normalising the Buy/Sell threshold by each stock's Average True Range,
we ensure that:
- Volatile stocks need larger moves to trigger a Buy/Sell signal.
- Low-volatility stocks get appropriate, tighter thresholds.

### Why market context features (Nifty 50, VIX)?

Individual stocks don't trade in a vacuum. ~60-70% of daily stock returns
in India are explained by broad market movement (beta). By giving our models
access to:
- Nifty 50 returns (what is the market doing?)
- India VIX (how scared are investors?)
- Sector indices (is your sector outperforming?)

...we let them learn conditional relationships: "this stock tends to
outperform when VIX is falling" or "this bank stock follows NIFTYBANK more
than NIFTY50."

### Why cyclical encoding (sin/cos) for time?

Day-of-week and month-of-year are cyclical: Monday (0) is close to Friday (4),
December (12) is close to January (1). If we use raw integers, the model sees
Monday and Friday as far apart.

sin/cos encoding preserves the circular relationship:
```
day_sin = sin(2π × dayofweek / 5)
day_cos = cos(2π × dayofweek / 5)
```

---

## 6. Ensemble Weighting Strategy

### Why different weights per timeframe?

**Short-term (7d, 15d):**
- Price momentum and technical indicators dominate.
- News sentiment has immediate impact.
- Fundamentals don't move in 7 days.
→ Heavy on CatBoost (25%) + LightGBM (20%) + Sentiment (15%).

**Medium-term (1m, 3m):**
- Trend and seasonality become important (Prophet's strength).
- All models contribute relatively equally.
→ Balanced, with Prophet slightly higher (20%).

**Long-term (6m, 1y, 3y):**
- Fundamentals drive long-term returns (P/E reversion, earnings growth).
- Technical signals are mostly noise at 1-year horizons.
→ Fundamentals dominate at 40%.

This is based on well-established financial research:
- Short-term: momentum factor (Jegadeesh & Titman, 1993)
- Long-term: value factor (Fama & French, 1993)

---

## 7. Validation Methodology

### Why purged walk-forward validation?

Standard k-fold cross-validation is **illegal** in time-series:
- It randomly mixes past and future data.
- Your model trains on 2024 data and predicts 2023 data.
- Accuracy looks amazing but is completely fake.

**Walk-forward** fixes this: always train on past, predict future.

**Purging** goes further: after the training set ends, we skip
`horizon_days + 5 days` before the test set starts. This prevents:
- **Label leakage**: If predicting 30-day returns, the last 30 days of
  training have targets that overlap with test labels.
- **Autocorrelation leakage**: Today's features are correlated with
  yesterday's features. A small gap breaks this correlation.

This technique is from Marcos López de Prado's "Advances in Financial
Machine Learning" — the gold standard in quant finance.

---

## 8. Confidence Calibration

Raw model probabilities (e.g., CatBoost says "72% Buy") are often
**overconfident**. Financial models in particular tend to be miscalibrated
because the training distribution shifts constantly.

Our calibration has three components:

1. **Agreement bonus (+10%)**: If 4+ out of 6 models agree on direction,
   they're probably right. Collective intelligence > individual model.

2. **Disagreement penalty (cap at 40%)**: If models are split 50/50,
   nobody really knows. Capping confidence communicates this honestly.

3. **Low-data penalty (-20%)**: Stocks with <2 years of history have
   insufficient training data. We reduce confidence to flag this.

---

## 9. Fundamental Scoring System

### Why 5 factors instead of a single metric?

A single metric (e.g., P/E ratio) can be misleading:
- Low P/E might mean undervalued OR earnings are about to collapse.
- High ROE might mean great management OR excessive leverage.

By scoring across 5 orthogonal factors, we get a balanced view:
- **Valuation** (25%): Are you paying a fair price? (P/E, P/B, PEG)
- **Profitability** (25%): Does the business make money? (ROE, margins)
- **Financial Health** (20%): Can it survive a downturn? (D/E, current ratio)
- **Growth** (20%): Is it expanding? (revenue, earnings growth)
- **Dividend** (10%): Does it return cash? (yield, payout ratio)

### Why absolute thresholds instead of sector-relative?

Ideally, we'd compare each stock to its sector peers. But yfinance doesn't
provide bulk peer data for free. So we use broad-market benchmarks that
work reasonably well across Indian equities (e.g., P/E 5-40 range covers
everything from PSU banks to IT companies).

---

## 10. Sentiment Analysis Pipeline

### Google News RSS → FinBERT

**Step 1: Fetch headlines**
- Google News RSS is free, no API key, no rate limits.
- We search for `"{company name} stock"` in Indian English (hl=en-IN).
- Top 15 headlines provide a snapshot of recent sentiment.

**Step 2: FinBERT classification**
- Each headline is classified as positive / negative / neutral.
- Net score = P(positive) - P(negative) per headline.
- Aggregate = average of all net scores.

**Step 3: Signal derivation**
- Score > +0.15 → Bullish
- Score < -0.15 → Bearish
- Otherwise → Neutral

The ±0.15 threshold avoids noise — many headlines are neutral factual
statements that shouldn't move the needle.

---

## 11. UI/UX Design Choices

### Dark glassmorphism theme

**Why dark?** Finance professionals and retail traders overwhelmingly prefer
dark interfaces (Bloomberg Terminal, TradingView, Zerodha Kite are all dark).

**Why glassmorphism?** Modern, premium feel with subtle depth. The frosted-glass
cards create visual hierarchy without harsh borders.

**Color system:**
- Teal (#00D4AA) — primary accent (trust, technology)
- Green (#00E676) — Buy signals
- Red (#FF5252) — Sell signals
- Amber (#FFB74D) — Hold signals
- Navy (#0E1117) — background

### Inter font

Chosen because it's:
- Designed specifically for screens (not print)
- Excellent readability at small sizes (metric labels)
- Available free via Google Fonts

---

## 12. PDF Report Generation

### fpdf2 over alternatives

**Chosen over:** ReportLab, WeasyPrint, xhtml2pdf

- **fpdf2**: Lightweight, no system dependencies, simple API. Perfect for
  generating structured reports with text + images.
- **ReportLab**: More powerful but heavier and has a steeper learning curve.
- **WeasyPrint**: HTML→PDF conversion, but requires system-level dependencies
  (Cairo, Pango) that are painful on Windows.

### Chart export via kaleido

Plotly charts are HTML/JavaScript — they can't go directly into PDFs. The
`kaleido` engine (by the Plotly team) converts them to static PNG/SVG. We
render at 2x scale for print-quality resolution.

---

## 13. Performance & Caching

### Caching strategy

| What | Decorator | TTL | Why |
|------|-----------|-----|-----|
| Historical data | `@st.cache_data` | 1 hour | EOD data doesn't change intraday |
| Fundamentals | `@st.cache_data` | 1 hour | Same as above |
| Market data | `@st.cache_data` | 1 hour | Nifty/VIX data |
| FinBERT model | `@st.cache_resource` | Forever | 1.3GB model, load once |
| Chronos model | `@st.cache_resource` | Forever | 200MB model, load once |

### Expected runtime

| Step | Time |
|------|------|
| Ticker validation | ~2s |
| Historical data fetch | ~3s |
| Market context | ~5s |
| Technical indicators | <1s |
| Fundamentals | ~2s |
| Sentiment (15 headlines) | ~3s |
| CatBoost training + prediction | ~5s |
| LightGBM training + prediction | ~3s |
| Prophet (7 timeframes) | ~30s |
| LSTM inference | <1s |
| Chronos inference (7 timeframes) | ~10s |
| **Total** | **~65s** |

---

## 14. Limitations & Honest Disclaimers

1. **This is NOT a crystal ball.** No model can reliably predict stock prices.
   Markets are partially efficient and incorporate information faster than any
   model can process it.

2. **Long-term predictions (1y, 3y) are speculative.** The confidence scores
   reflect this — they will be low. The confidence gating may hide these
   timeframes entirely.

3. **yfinance data quality**: Some fundamental fields may be stale or missing.
   Always verify critical decisions against official sources (NSE/BSE websites).

4. **News sentiment is headline-only**: FinBERT analyses headlines, not full
   articles. Headlines can be misleading or sensationalised.

5. **No FII/DII data**: Foreign and domestic institutional investor flows are
   strong predictors but not available for free in a clean API.

6. **Survivorship bias**: We train on stocks that exist today. Stocks that
   delisted or went bankrupt are not in the training data, which biases
   predictions slightly upward.

7. **The models retrain on each request**: CatBoost and LightGBM train fresh
   each time. This means the model is always "current" but predictions may
   vary slightly between runs due to randomness in training.
