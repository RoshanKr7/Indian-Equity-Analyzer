# AI Indian Equity Predictor

An ensemble machine learning & deep learning web application for analyzing and forecasting Indian equities (NSE & BSE) across multiple timeframes.

---

## Key Features

- **Multi-Model Forecasting Ensemble**:
  - **Gradient Boosting**: XGBoost (Temporal Histogram) & LightGBM (DART Boosting).
  - **Deep Time-Series**: Neural Hierarchical Interpolation (N-HiTS) & Temporal Fusion Transformer (TFT).
  - **Foundation Models**: Zero-shot time-series forecasting via Amazon Chronos.
  - **Fallback Models**: BiLSTM Attention & Prophet.
- **Fundamental & Technical Analysis**:
  - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, Volatility).
  - Fundamental scoring system (P/E, ROE, Debt/Equity, Earnings Growth).
  - Benchmark comparison vs NIFTY 50 and Sectoral Indices.
- **Sentiment Pipeline**:
  - Real-time Google News RSS scraping and sentiment scoring using FinancialBERT fine-tuned for Indian financial context.
- **Interactive UI & Reporting**:
  - Built with Streamlit and Plotly interactive charts.
  - Automated executive PDF report generation.

---

## Project Structure

```
.
├── app.py                      # Main Streamlit web application entry point
├── config/                     # Configuration and hyperparameters
│   └── settings.py             # Global constants, weights, and timeframe definitions
├── core/                       # Core analytical engines
│   ├── data_fetcher.py         # yfinance data ingestion & caching
│   ├── fundamental_analysis.py # Valuation & financial ratio scoring
│   ├── market_context.py       # NIFTY index & sectoral benchmark analysis
│   ├── sentiment_analysis.py   # News scraping & sentiment scoring
│   ├── technical_analysis.py   # Technical indicators computation
│   └── ticker_validator.py     # NSE/BSE ticker resolution & validation
├── models/                     # ML/DL forecasting architectures
│   ├── ensemble.py             # Dynamic multi-model weighted ensemble
│   ├── feature_engineer.py     # Lag features, rolling statistics, calendar signals
│   ├── xgboost_temporal_model.py
│   ├── lightgbm_model.py
│   ├── nhits_model.py
│   ├── tft_model.py
│   ├── chronos_model.py
│   ├── lstm_model.py
│   ├── prophet_model.py
│   └── pretrained/             # Model weights and configurations
├── ui/                         # Presentation layer
│   ├── charts.py               # Plotly charting functions
│   ├── components.py           # Streamlit UI widgets & metric cards
│   ├── report_generator.py     # PDF export engine (fpdf2)
│   └── styles.css              # Custom dashboard styling
├── requirements.txt            # Project dependencies
└── README.md
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd "Stock Analysis"
```

### 2. Set up Python Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## Disclaimer

This application is built for educational and research purposes only. It is not financial advice. Indian equity markets carry financial risk; always do your own due diligence before making investment decisions.
