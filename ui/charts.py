"""
Plotly chart builders for the analysis dashboard.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def candlestick_with_volume(df: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
    """Interactive candlestick chart with volume subplot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00E676", decreasing_line_color="#FF5252",
        increasing_fillcolor="#00E676", decreasing_fillcolor="#FF5252",
        name="Price",
    ), row=1, col=1)

    # Volume bars colored by direction
    colors = ["#00E676" if c >= o else "#FF5252"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, opacity=0.5, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=30),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def technical_overlay(df: pd.DataFrame) -> go.Figure:
    """Price with SMA/EMA/Bollinger Band overlay."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Close",
        line=dict(color="#E2E8F0", width=1.5),
    ))

    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"], name="SMA 50",
            line=dict(color="#00D4AA", width=1, dash="dash"),
        ))
    if "SMA_200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_200"], name="SMA 200",
            line=dict(color="#FF9100", width=1, dash="dash"),
        ))
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(0,150,255,0.3)", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(0,150,255,0.3)", width=1),
            fill="tonexty", fillcolor="rgba(0,150,255,0.05)",
        ))

    fig.update_layout(
        title="Technical Overlay",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        height=450,
        margin=dict(l=50, r=20, t=50, b=30),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI with overbought/oversold zones."""
    fig = go.Figure()

    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI_14"], name="RSI (14)",
            line=dict(color="#00D4AA", width=2),
        ))
        # Overbought / oversold zones
        fig.add_hline(y=70, line_dash="dot", line_color="#FF5252",
                      annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dot", line_color="#00E676",
                      annotation_text="Oversold (30)")
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,82,82,0.08)", line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,230,118,0.08)", line_width=0)

    fig.update_layout(
        title="RSI (14)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        height=300,
        yaxis_range=[0, 100],
        margin=dict(l=50, r=20, t=50, b=30),
    )
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD histogram with signal line."""
    fig = go.Figure()

    if "MACD_Hist" in df.columns:
        colors = ["#00E676" if v >= 0 else "#FF5252"
                  for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"],
            marker_color=colors, name="Histogram", opacity=0.6,
        ))
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#0096FF", width=1.5),
        ))
    if "MACD_Signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#FF9100", width=1.5),
        ))

    fig.update_layout(
        title="MACD",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        height=300,
        margin=dict(l=50, r=20, t=50, b=30),
    )
    return fig


def sentiment_bar_chart(details: list) -> go.Figure:
    """Horizontal bar chart of per-headline sentiment scores."""
    if not details:
        return go.Figure()

    titles = [d.get("title", "")[:60] + "..." for d in details]
    scores = [d.get("net_score", 0) for d in details]
    colors = ["#00E676" if s > 0 else "#FF5252" if s < 0 else "#FFB74D"
              for s in scores]

    fig = go.Figure(go.Bar(
        y=titles, x=scores,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="Headline Sentiment Scores",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        height=max(300, len(details) * 35),
        margin=dict(l=300, r=20, t=50, b=30),
        xaxis_title="Sentiment Score",
    )
    return fig
