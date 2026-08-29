"""
Indian Equity Analyzer — Main Streamlit Application.

Flow:
  1. User enters ticker → Validate (NSE first, BSE fallback)
  2. Show company info → User confirms
  3. Run analysis pipeline (5 steps with progress)
  4. Display results in tabs (Charts, Fundamentals, Sentiment, Predictions)
  5. Generate & download PDF report

AI Toggle:
  Sidebar switch to enable/disable all ML predictions (default: ON).
  When OFF, only technical indicators and fundamentals drive signals.
"""

import os
import warnings

# Suppress noisy HuggingFace docstring/internal validation logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

import streamlit as st

from config.settings import APP_TITLE, APP_ICON, TIMEFRAMES
from core.ticker_validator import validate_ticker
from core.data_fetcher import fetch_historical, fetch_fundamentals
from core.market_context import fetch_market_data, fetch_sector_index, compute_market_features
from core.fundamental_analysis import compute_factor_scores, compute_composite_score
from core.technical_analysis import compute_all_indicators, compute_technical_score
from core.sentiment_analysis import fetch_google_news, analyze_sentiment
from models.ensemble import EnsemblePredictor
from models.lstm_model import LSTMPredictor
from config.settings import AI_ENABLED_DEFAULT
from ui.components import (
    render_company_header, render_prediction_dashboard,
    render_fundamental_grid, render_sentiment_results,
    render_model_breakdown, render_disclaimer,
)
from ui.charts import (
    candlestick_with_volume, technical_overlay,
    rsi_chart, macd_chart, sentiment_bar_chart,
)
from ui.report_generator import ReportGenerator


# ── Page Config ──
st.set_page_config(
    page_title="Indian Equity Analyzer",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ──
css_path = os.path.join(os.path.dirname(__file__), "ui", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown(f"# {APP_TITLE}")
    st.markdown("---")

    ticker_input = st.text_input(
        "Enter Stock Ticker",
        placeholder="e.g., RELIANCE, TCS, INFY",
        help="Enter the NSE/BSE ticker symbol without exchange suffix",
    )

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.markdown("---")

    # ── AI Toggle ──
    st.markdown("### ⚙️ AI Settings")
    ai_enabled = st.toggle(
        "🤖 AI Predictions",
        value=AI_ENABLED_DEFAULT,
        help="Enable or disable all machine learning predictions. "
             "When OFF, signals are derived from technical indicators "
             "and fundamentals only.",
        key="ai_toggle",
    )

    enhanced_sentiment = False
    if ai_enabled:
        enhanced_sentiment = st.toggle(
            "🧠 Enhanced Sentiment (Qwen AI)",
            value=False,
            help="Use Qwen2.5-1.5B-Instruct for deeper sentiment reasoning. "
                 "First-run requires ~3GB download. Adds 3-5 seconds per analysis.",
            key="enhanced_sentiment_toggle",
        )

    st.markdown("---")
    if ai_enabled:
        st.markdown("""
        **How it works:**
        1. Enter any Indian stock ticker
        2. Confirm the company
        3. Get AI-powered analysis & predictions

        **Models (always-on):**
        - ⚡ XGBoost 3.2
        - 🌿 LightGBM DART
        - 🔁 XGBoost Temporal (sequential)
        - ⏱️ Amazon Chronos-Bolt

        **Models (optional):**
        - 🧠 TFT (Temporal Fusion Transformer)
        - 📈 N-HiTS (replaces Prophet, 50x faster)
        - 🤖 BiLSTM+Attention (fallback for TFT)

        **Sentiment:**
        - 🇮🇳 FinancialBERT-Indian (if fine-tuned)
        - 📰 FinancialBERT-2023 (default)
        """)
    else:
        st.markdown("""
        **AI Predictions: OFF**

        Running in Technical Analysis mode:
        - 📊 RSI, MACD, Bollinger Bands
        - 📋 Fundamental factor scores
        - 📰 Keyword-based news sentiment

        *Enable AI toggle for ML predictions.*
        """)


# ── BiLSTM Check (informational only — XGBoost Temporal covers when absent) ──
lstm_checker = LSTMPredictor()
lstm_available = lstm_checker.check_weights_exist()


# ── Session State ──
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "company_info" not in st.session_state:
    st.session_state.company_info = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ── Main Content ──
if not lstm_available:
    st.info("""
    💡 **BiLSTM+Attention weights not found** — running with XGBoost Temporal as sequential model.
    For maximum accuracy, train the upgraded BiLSTM on Kaggle GPU:
    1. Open `kaggle/train_bilstm_attention.ipynb` in [Kaggle](https://www.kaggle.com)
    2. Enable GPU: Settings → Accelerator → GPU T4 x2 or P100
    3. Run all cells (~20-30 min)
    4. Download: `bilstm_weights.pth`, `feature_scaler.pkl`, `model_config.json`
    5. Place in `models/pretrained/`

    *The app works fully without these weights. XGBoost Temporal provides sequential predictions.*
    """)
    st.markdown("---")

# Title
st.markdown(f"<h1 style='text-align: center; color: #00D4AA;'>{APP_TITLE}</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #718096;'>AI-Powered Stock Analysis & Prediction for Indian Equities</p>",
            unsafe_allow_html=True)

# ── Step 1: Validate Ticker ──
if analyze_btn and ticker_input:
    st.session_state.confirmed = False
    st.session_state.analysis_done = False

    with st.spinner("Validating ticker..."):
        info = validate_ticker(ticker_input.strip())

    if info is None:
        st.error(f"❌ Could not find **{ticker_input.upper()}** on NSE or BSE. "
                 "Please check the ticker and try again.")
    else:
        st.session_state.company_info = info
        st.session_state.waiting_confirmation = True

# ── Step 2: Confirm ──
if (st.session_state.company_info is not None
        and not st.session_state.confirmed
        and st.session_state.get("waiting_confirmation", False)):

    info = st.session_state.company_info
    render_company_header(info)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, analyze this stock", type="primary",
                      use_container_width=True):
            st.session_state.confirmed = True
            st.session_state.waiting_confirmation = False
            st.rerun()
    with col2:
        if st.button("❌ No, try another", use_container_width=True):
            st.session_state.company_info = None
            st.session_state.waiting_confirmation = False
            st.rerun()

# ── Step 3: Full Analysis ──
if st.session_state.confirmed and not st.session_state.analysis_done:
    info = st.session_state.company_info
    render_company_header(info)

    progress = st.progress(0, text="Starting analysis...")

    # 1. Historical data
    progress.progress(10, text="📊 Fetching historical data...")
    stock_df = fetch_historical(info["symbol"])
    if stock_df.empty:
        st.error("Failed to fetch historical data. Please try again.")
        st.stop()

    # 2. Market context
    progress.progress(25, text="🌍 Loading market context (Nifty 50, VIX)...")
    market_data = fetch_market_data()
    sector_index = fetch_sector_index(info.get("sector", ""))
    market_features = compute_market_features(stock_df, market_data, info.get("sector"), sector_index)

    # 3. Technical indicators
    progress.progress(40, text="📈 Computing technical indicators...")
    stock_df = compute_all_indicators(stock_df)
    tech_score = compute_technical_score(stock_df)

    # 4. Fundamentals
    progress.progress(50, text="📋 Analyzing fundamentals...")
    fundamentals = fetch_fundamentals(info["symbol"])
    factor_scores = compute_factor_scores(fundamentals)
    composite_score = compute_composite_score(factor_scores)

    # 5. Sentiment
    sent_label = "📰 Scanning news sentiment (FinancialBERT-2023)..."
    if enhanced_sentiment:
        sent_label = "📰 Scanning news sentiment (Qwen AI Enhanced)..."
    progress.progress(65, text=sent_label)
    company_name = info.get("long_name", info.get("short_name", ""))
    headlines = fetch_google_news(company_name)
    sentiment = analyze_sentiment(headlines, enhanced_mode=enhanced_sentiment)

    # 6. Predictions
    if ai_enabled:
        progress.progress(80, text="🤖 Running AI prediction ensemble...")
    else:
        progress.progress(80, text="📊 Computing technical signals (AI OFF)...")
    current_price = float(stock_df["Close"].iloc[-1])
    ensemble = EnsemblePredictor()
    predictions = ensemble.predict_all_timeframes(
        stock_df, market_features,
        composite_score, sentiment,
        current_price,
        ai_enabled=ai_enabled,
    )

    progress.progress(100, text="✅ Analysis complete!")

    # Store results
    st.session_state.stock_df = stock_df
    st.session_state.market_features = market_features
    st.session_state.fundamentals = fundamentals
    st.session_state.factor_scores = factor_scores
    st.session_state.composite_score = composite_score
    st.session_state.tech_score = tech_score
    st.session_state.sentiment = sentiment
    st.session_state.predictions = predictions
    st.session_state.current_price = current_price
    st.session_state.analysis_done = True
    st.session_state.ai_enabled_used = ai_enabled
    st.rerun()

# ── Step 4: Display Results ──
if st.session_state.analysis_done:
    info = st.session_state.company_info
    stock_df = st.session_state.stock_df
    fundamentals = st.session_state.fundamentals
    factor_scores = st.session_state.factor_scores
    composite_score = st.session_state.composite_score
    sentiment = st.session_state.sentiment
    predictions = st.session_state.predictions
    current_price = st.session_state.current_price

    render_company_header(info)

    # Current price display
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 0.85rem; color: #A0AEC0;">Current Price</div>
        <div style="font-size: 2.5rem; font-weight: 700; color: #FFFFFF;">
            ₹{current_price:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI status banner
    ai_was_on = st.session_state.get("ai_enabled_used", True)
    preds = st.session_state.predictions
    sample_tf = next(iter(preds.values())) if preds else {}
    tft_avail   = sample_tf.get("tft_available", False)
    nhits_avail = sample_tf.get("nhits_available", False)
    lstm_avail  = sample_tf.get("lstm_available", False)

    if not ai_was_on:
        st.warning(
            "🔌 **AI Predictions are OFF** — showing technical & fundamental signals only. "
            "Enable the AI toggle in the sidebar for full ML predictions."
        )
    else:
        seq_model  = "✅ TFT" if tft_avail else ("✅ BiLSTM" if lstm_avail else "⚡ XGBoost Temporal")
        trend_model = "✅ N-HiTS" if nhits_avail else "🔄 Prophet (fallback)"
        sent_note   = "🇮🇳" if os.path.exists("models/pretrained/finbert_indian") else "📰"
        st.success(
            f"🤖 AI Active | Sequential: {seq_model} | Trend: {trend_model} | Sentiment: {sent_note}"
        )

    # Tabs
    pred_tab_label = "🤖 AI Predictions" if ai_was_on else "📊 Technical Signals"
    tab_charts, tab_fund, tab_sent, tab_pred, tab_download = st.tabs([
        "📊 Price & Technicals",
        "📋 Fundamentals",
        "📰 Sentiment",
        pred_tab_label,
        "📥 Download Report",
    ])

    # ── Charts Tab ──
    with tab_charts:
        # Show last 1 year by default
        df_1y = stock_df.tail(252)
        st.plotly_chart(candlestick_with_volume(df_1y, f"{info['short_name']} — 1 Year"),
                        use_container_width=True)
        st.plotly_chart(technical_overlay(df_1y), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(rsi_chart(df_1y), use_container_width=True)
        with col2:
            st.plotly_chart(macd_chart(df_1y), use_container_width=True)

    # ── Fundamentals Tab ──
    with tab_fund:
        render_fundamental_grid(fundamentals, factor_scores)
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #A0AEC0;">Composite Fundamental Score</div>
            <div style="font-size: 2rem; font-weight: 700; color: {'#00E676' if composite_score > 0.6 else '#FFB74D' if composite_score > 0.4 else '#FF5252'};">
                {composite_score*100:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        if fundamentals.get("description"):
            with st.expander("About the Company"):
                st.write(fundamentals["description"])

    # ── Sentiment Tab ──
    with tab_sent:
        # Show which sentiment model was used
        sent_model = sentiment.get("model", "FinancialBERT")
        if sent_model and "Qwen" in sent_model:
            st.info(f"🧠 **Enhanced Sentiment** powered by {sent_model}")
        elif sent_model:
            st.info(f"📰 **Sentiment Model:** {sent_model}")
        render_sentiment_results(sentiment)
        if sentiment.get("details"):
            st.plotly_chart(
                sentiment_bar_chart(sentiment["details"]),
                use_container_width=True,
            )

    # ── Predictions Tab ──
    with tab_pred:
        render_prediction_dashboard(predictions)

        st.markdown("---")
        st.markdown("### 🔍 Detailed Model Breakdown")
        selected_tf = st.selectbox(
            "Select timeframe for breakdown:",
            options=list(predictions.keys()),
            format_func=lambda x: predictions[x]["timeframe"],
        )
        if selected_tf:
            render_model_breakdown(predictions[selected_tf])

    # ── Download Tab ──
    with tab_download:
        st.markdown("### 📥 Download Full Analysis Report")
        st.markdown("Generate a comprehensive PDF report with all charts, metrics, and predictions.")

        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                df_1y = stock_df.tail(252)
                chart_figs = {
                    "candlestick": candlestick_with_volume(df_1y, f"{info['short_name']}"),
                    "technical_overlay": technical_overlay(df_1y),
                    "rsi": rsi_chart(df_1y),
                    "macd": macd_chart(df_1y),
                }

                report_gen = ReportGenerator()
                pdf_bytes = report_gen.generate(
                    company_info=info,
                    fundamentals=fundamentals,
                    factor_scores=factor_scores,
                    composite_score=composite_score,
                    sentiment=sentiment,
                    predictions=predictions,
                    charts=chart_figs,
                )

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{info['symbol']}_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("Report generated successfully!")

    # Disclaimer
    render_disclaimer()
