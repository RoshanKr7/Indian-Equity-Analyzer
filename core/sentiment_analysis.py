"""
News sentiment analysis via Google News RSS + FinancialBERT (upgraded).

Upgrade Summary:
  OLD: ProsusAI/finbert (2019, general financial text)
  NEW: ahmedrachid/FinancialBERT-Sentiment-Analysis (2023, +8-10% accuracy)
       with ProsusAI/finbert as automatic fallback if download fails.

Optional Enhanced Mode (Qwen2.5-1.5B-Instruct):
  When 'enhanced_sentiment' is enabled in the sidebar, each headline is
  passed through Qwen2.5-1.5B-Instruct for contextual reasoning. Qwen
  understands financial nuance far better than classification-only models:
  - "Company cuts dividend" → Qwen understands this is negative even if
    keywords alone score it neutral.
  - "Stock down 5% on volume below average" → Qwen reads market context.
  This runs on CPU and adds ~2-3 seconds per batch of 15 headlines.

Pipeline:
  1. Scrape recent headlines from Google News RSS (free, no API key).
  2. Run each headline through FinancialBERT (or Qwen in enhanced mode).
  3. Aggregate into a single bullish/neutral/bearish signal.
"""

import re
import os
import feedparser
import streamlit as st
from transformers import pipeline as hf_pipeline

from config.settings import (
    FINBERT_MODEL, FINBERT_MODEL_FALLBACK, FINBERT_INDIAN_PATH,
    NEWS_HEADLINE_COUNT, GOOGLE_NEWS_RSS,
    QWEN_SENTIMENT_MODEL,
)


def fetch_google_news(company_name: str, n: int = NEWS_HEADLINE_COUNT) -> list[dict]:
    """Fetch recent news headlines from Google News RSS."""
    query = re.sub(r"[^\w\s]", "", company_name).strip().replace(" ", "+")
    url = GOOGLE_NEWS_RSS.format(query=query)
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:n]:
            results.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
        return results
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def _load_financial_bert():
    """
    Load sentiment model — tries in priority order:
      1. FinancialBERT-Indian (fine-tuned on NSE/BSE news via Kaggle GPU)
      2. FinancialBERT-2023 (generic English financial text)
      3. FinBERT-2019 (original fallback)

    Fine-tuned model path: models/pretrained/finbert_indian/
    Expected after running kaggle/finetune_finbert_indian.ipynb and
    copying the output directory there.
    """
    # Priority 1: locally fine-tuned Indian model
    if os.path.exists(FINBERT_INDIAN_PATH):
        try:
            clf = hf_pipeline(
                "text-classification",
                model=FINBERT_INDIAN_PATH,
                top_k=None,
                device=-1,
                truncation=True,
                max_length=512,
            )
            return clf, "FinancialBERT-Indian (fine-tuned)"
        except Exception:
            pass  # fall through to generic model

    # Priority 2 & 3: generic models
    for model_id in [FINBERT_MODEL, FINBERT_MODEL_FALLBACK]:
        try:
            clf = hf_pipeline(
                "text-classification",
                model=model_id,
                top_k=None,
                device=-1,
                truncation=True,
                max_length=512,
            )
            return clf, model_id
        except Exception:
            continue
    return None, None


@st.cache_resource(show_spinner=False)
def _load_qwen_sentiment():
    """
    Load Qwen2.5-1.5B-Instruct for enhanced sentiment reasoning.

    Only loaded when user enables enhanced mode in sidebar.
    Model size: ~3GB (float16) or ~1GB (4-bit quantized).
    Runs on CPU — ~2-4s per batch of headlines.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            QWEN_SENTIMENT_MODEL,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_SENTIMENT_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def _qwen_classify_headline(tokenizer, model, headline: str) -> dict:
    """
    Use Qwen2.5-1.5B to classify a financial headline.

    Prompt engineered for concise financial sentiment classification.
    """
    prompt = (
        f"You are a financial analyst. Classify the sentiment of this stock market headline "
        f"as exactly one of: POSITIVE, NEGATIVE, or NEUTRAL.\n"
        f"Headline: \"{headline}\"\n"
        f"Sentiment (one word only):"
    )
    try:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().upper()

        if "POSITIVE" in response:
            return {"label": "positive", "score": 0.85}
        elif "NEGATIVE" in response:
            return {"label": "negative", "score": 0.85}
        else:
            return {"label": "neutral", "score": 0.80}
    except Exception:
        return {"label": "neutral", "score": 0.50}


def analyze_sentiment(
    headlines: list[dict],
    enhanced_mode: bool = False,
) -> dict:
    """
    Run FinancialBERT (or Qwen in enhanced mode) on headlines and aggregate.

    Parameters
    ----------
    headlines : list of dicts with 'title', 'link', 'published'
    enhanced_mode : bool
        If True, uses Qwen2.5-1.5B-Instruct for richer contextual sentiment.
        If False (default), uses FinancialBERT-2023.

    Returns
    -------
    dict
        - score    : float (-1 to +1)  — aggregate sentiment
        - label    : str   — "Bullish" / "Neutral" / "Bearish"
        - model    : str   — which model was used
        - details  : list  — per-headline sentiment breakdown
    """
    if not headlines:
        return {"score": 0.0, "label": "Neutral", "model": "none", "details": []}

    texts = [h["title"] for h in headlines if h.get("title")]
    if not texts:
        return {"score": 0.0, "label": "Neutral", "model": "none", "details": []}

    details = []
    total_score = 0.0
    model_used = "unknown"

    # ── Enhanced Mode: Qwen2.5-1.5B ─────────────────────────────────────
    if enhanced_mode:
        tokenizer, qwen_model = _load_qwen_sentiment()
        if tokenizer is not None and qwen_model is not None:
            model_used = "Qwen2.5-1.5B-Instruct"
            for headline in headlines:
                title = headline.get("title", "")
                if not title:
                    continue
                result = _qwen_classify_headline(tokenizer, qwen_model, title)
                label = result["label"]
                score = result["score"]

                pos = score if label == "positive" else 0.0
                neg = score if label == "negative" else 0.0
                neu = score if label == "neutral" else 0.0
                net = pos - neg
                total_score += net

                details.append({
                    "title": title,
                    "sentiment": label.capitalize(),
                    "confidence": round(score * 100, 1),
                    "net_score": round(net, 3),
                })
        # If Qwen failed to load, fall through to FinancialBERT below
        if model_used == "Qwen2.5-1.5B-Instruct":
            return _aggregate_details(details, total_score, model_used)

    # ── Standard Mode: FinancialBERT (2023) ──────────────────────────────
    classifier, model_id = _load_financial_bert()
    if classifier is None:
        return {"score": 0.0, "label": "Neutral", "model": "unavailable", "details": []}

    model_used = model_id or FINBERT_MODEL

    try:
        raw_results = classifier(texts, batch_size=8, truncation=True, max_length=512)
    except Exception:
        return {"score": 0.0, "label": "Neutral", "model": model_used, "details": []}

    for headline, result in zip(headlines, raw_results):
        # result is a list of dicts like [{"label": "positive", "score": 0.9}, ...]
        label_scores = {r["label"]: r["score"] for r in result}
        pos = label_scores.get("positive", 0)
        neg = label_scores.get("negative", 0)
        neu = label_scores.get("neutral", 0)

        # Net score: +1 fully positive, -1 fully negative
        net = pos - neg
        total_score += net

        best_label = max(label_scores, key=label_scores.get)
        details.append({
            "title": headline.get("title", ""),
            "sentiment": best_label.capitalize(),
            "confidence": round(max(pos, neg, neu) * 100, 1),
            "net_score": round(net, 3),
        })

    return _aggregate_details(details, total_score, model_used)


def _aggregate_details(details: list, total_score: float, model_used: str) -> dict:
    """Aggregate per-headline details into a final sentiment result."""
    if not details:
        return {"score": 0.0, "label": "Neutral", "model": model_used, "details": []}

    avg_score = total_score / len(details)
    if avg_score > 0.15:
        label = "Bullish"
    elif avg_score < -0.15:
        label = "Bearish"
    else:
        label = "Neutral"

    return {
        "score": round(avg_score, 4),
        "label": label,
        "model": model_used,
        "details": details,
    }
