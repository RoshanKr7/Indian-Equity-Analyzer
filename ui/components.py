"""
Reusable Streamlit UI components.

All HTML-heavy rendering lives here to keep app.py clean.
"""

import streamlit as st
from core.ticker_validator import format_market_cap
from config.settings import DISCLAIMER


def render_company_header(info: dict):
    """Large company card with name, sector, exchange badge."""
    badge_class = "badge-nse" if info["exchange"] == "NSE" else "badge-bse"
    mcap = format_market_cap(info.get("market_cap"))

    st.markdown(f"""
    <div class="company-header">
        <div class="company-name">{info['long_name']}</div>
        <div class="company-meta">
            <span class="badge {badge_class}">{info['exchange']}</span>
            &nbsp; {info['symbol']} &nbsp;·&nbsp;
            {info.get('sector', 'N/A')} &nbsp;·&nbsp;
            {info.get('industry', 'N/A')} &nbsp;·&nbsp;
            Market Cap: {mcap}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None, delta_dir: str = ""):
    """Styled metric card with optional delta."""
    delta_html = ""
    if delta:
        css = "metric-delta-up" if delta_dir == "up" else "metric-delta-down"
        arrow = "↑" if delta_dir == "up" else "↓"
        delta_html = f'<div class="{css}">{arrow} {delta}</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_prediction_card(tf_label: str, result: dict):
    """Single prediction card with signal, price, confidence."""
    signal = result["signal"]
    confidence = result["confidence"]
    pred_price = result.get("predicted_price", "N/A")
    pred_return = result.get("predicted_return", 0)
    current_price = result.get("current_price", 0)
    gated = result.get("gated", False)

    signal_class = f"signal-{signal.lower()}"
    return_class = "prediction-return-up" if pred_return >= 0 else "prediction-return-down"
    return_sign = "+" if pred_return >= 0 else ""
    conf_pct = round(confidence * 100, 1)

    # Confidence bar color
    if conf_pct >= 60:
        bar_class = "conf-high"
    elif conf_pct >= 40:
        bar_class = "conf-med"
    else:
        bar_class = "conf-low"

    if gated:
        st.markdown(f"""
        <div class="prediction-card" style="opacity: 0.5;">
            <div class="prediction-timeframe">{tf_label}</div>
            <div style="color: #718096; font-size: 0.9rem; margin: 1rem 0;">
                ⚠️ Insufficient confidence<br>
                <span style="font-size: 0.75rem;">({conf_pct}% — below threshold)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-timeframe">{tf_label}</div>
            <div><span class="{signal_class}">{signal}</span></div>
            <div class="prediction-price">₹{pred_price:,.2f}</div>
            <div class="{return_class}">{return_sign}{pred_return*100:.1f}%</div>
            <div style="margin-top: 0.5rem;">
                <span style="font-size: 0.75rem; color: #A0AEC0;">
                    Confidence: {conf_pct}%
                </span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill {bar_class}"
                         style="width: {conf_pct}%;"></div>
                </div>
            </div>
            <div style="font-size: 0.7rem; color: #718096; margin-top: 0.3rem;">
                Current: ₹{current_price:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_dashboard(results: dict):
    """Grid layout of all prediction cards."""
    st.markdown("### 🤖 AI Prediction Dashboard")

    # Row 1: Short-term (7d, 15d)
    cols = st.columns(2)
    for i, tf_code in enumerate(["7d", "15d"]):
        if tf_code in results:
            with cols[i]:
                render_prediction_card(results[tf_code]["timeframe"], results[tf_code])

    # Row 2: Medium-term (1m, 3m)
    cols = st.columns(2)
    for i, tf_code in enumerate(["1m", "3m"]):
        if tf_code in results:
            with cols[i]:
                render_prediction_card(results[tf_code]["timeframe"], results[tf_code])

    # Row 3: Long-term (6m, 1y, 3y)
    cols = st.columns(3)
    for i, tf_code in enumerate(["6m", "1y", "3y"]):
        if tf_code in results:
            with cols[i]:
                render_prediction_card(results[tf_code]["timeframe"], results[tf_code])


def render_fundamental_grid(fundamentals: dict, factor_scores: dict):
    """Display fundamental metrics in a styled grid."""
    st.markdown("### 📋 Fundamental Analysis")

    # Factor scores bar
    for factor, data in factor_scores.items():
        score = data["score"]
        pct = round(score * 100)
        label = factor.replace("_", " ").title()
        bar_class = "conf-high" if pct >= 60 else ("conf-med" if pct >= 40 else "conf-low")
        st.markdown(f"""
        <div style="margin-bottom: 0.8rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <span style="color: #E2E8F0; font-weight: 500;">{label}</span>
                <span style="color: #A0AEC0;">{pct}%</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill {bar_class}" style="width: {pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics grid
    metrics = [
        ("P/E Ratio", fundamentals.get("pe_trailing"), None),
        ("P/B Ratio", fundamentals.get("pb_ratio"), None),
        ("ROE", fundamentals.get("roe"), "%"),
        ("Debt/Equity", fundamentals.get("debt_to_equity"), None),
        ("Profit Margin", fundamentals.get("profit_margin"), "%"),
        ("Revenue Growth", fundamentals.get("revenue_growth"), "%"),
        ("EPS", fundamentals.get("eps_trailing"), None),
        ("Dividend Yield", fundamentals.get("dividend_yield"), "%"),
    ]

    cols = st.columns(4)
    for i, (label, value, suffix) in enumerate(metrics):
        with cols[i % 4]:
            display_val = f"{value}{suffix}" if value is not None and suffix else (str(value) if value is not None else "N/A")
            render_metric_card(label, display_val)


def render_sentiment_results(sentiment: dict):
    """Display sentiment analysis results."""
    st.markdown("### 📰 News Sentiment Analysis")

    label = sentiment.get("label", "Neutral")
    score = sentiment.get("score", 0)

    color_map = {"Bullish": "#00E676", "Bearish": "#FF5252", "Neutral": "#FFB74D"}
    color = color_map.get(label, "#FFB74D")

    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 2rem; font-weight: 700; color: {color};">{label}</div>
        <div style="font-size: 1rem; color: #A0AEC0;">
            Aggregate Score: {score:+.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    details = sentiment.get("details", [])
    if details:
        for item in details:
            sent = item.get("sentiment", "Neutral")
            css_class = f"sentiment-{sent.lower()}"
            st.markdown(f"""
            <div style="padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span class="{css_class}">[{sent}]</span>
                &nbsp; {item.get('title', '')}
                <span style="color: #718096; font-size: 0.75rem;">
                    ({item.get('confidence', 0)}%)
                </span>
            </div>
            """, unsafe_allow_html=True)


def render_model_breakdown(result: dict):
    """Show per-model breakdown for a prediction."""
    components = result.get("component_results", {})
    if not components:
        return

    st.markdown("**Model Breakdown:**")
    for model, data in components.items():
        signal = data.get("signal", "Hold")
        conf = data.get("confidence", 0)
        st.markdown(f"- **{model.title()}**: {signal} ({conf*100:.0f}%)")


def render_disclaimer():
    """Render legal disclaimer."""
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
