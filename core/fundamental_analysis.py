"""
Multi-factor fundamental scoring.

Instead of a single metric we decompose fundamentals into five factors:
  1. Valuation      (25 %)  — Are you paying a fair price?
  2. Profitability   (25 %)  — Is the business earning well?
  3. Financial Health (20 %)  — Can it survive a downturn?
  4. Growth          (20 %)  — Is it expanding?
  5. Dividend        (10 %)  — Does it return cash to shareholders?

Each metric inside a factor is scored 0-1 on an absolute scale (using
thresholds typical for the Indian market).  A sector-relative adjustment
would be ideal but requires peer data which yfinance doesn't provide in
bulk for free — so we use broad-market benchmarks.
"""

from config.settings import FUNDAMENTAL_WEIGHTS


def _score_between(value, low, high, invert=False) -> float:
    """
    Linear score mapping *value* from [low, high] → [0, 1].
    If *invert* is True, lower values score higher (e.g. P/E, Debt/Equity).
    Returns 0.5 if value is None (neutral).
    """
    if value is None:
        return 0.5
    clamped = max(low, min(high, value))
    score = (clamped - low) / (high - low) if high != low else 0.5
    return 1.0 - score if invert else score


def compute_factor_scores(fundamentals: dict) -> dict:
    """
    Compute per-factor scores (each 0-1) from raw fundamental data.

    Parameters
    ----------
    fundamentals : dict
        Output of ``data_fetcher.fetch_fundamentals()``.

    Returns
    -------
    dict
        Keys: factor name → {"score": float, "details": dict}
    """
    # ── Valuation ──
    pe_score = _score_between(fundamentals.get("pe_trailing"), 5, 40, invert=True)
    pb_score = _score_between(fundamentals.get("pb_ratio"), 0.5, 10, invert=True)
    peg_score = _score_between(fundamentals.get("peg_ratio"), 0, 3, invert=True)
    valuation = (pe_score * 0.4 + pb_score * 0.3 + peg_score * 0.3)

    # ── Profitability ──
    roe_score    = _score_between(fundamentals.get("roe"), 0, 30)
    roa_score    = _score_between(fundamentals.get("roa"), 0, 15)
    margin_score = _score_between(fundamentals.get("profit_margin"), 0, 30)
    eps_val = fundamentals.get("eps_trailing")
    eps_score = 0.7 if eps_val is not None and eps_val > 0 else 0.3
    profitability = (roe_score * 0.35 + roa_score * 0.2 + margin_score * 0.25 + eps_score * 0.2)

    # ── Financial Health ──
    de_score = _score_between(fundamentals.get("debt_to_equity"), 0, 200, invert=True)
    cr_score = _score_between(fundamentals.get("current_ratio"), 0.5, 3)
    qr_score = _score_between(fundamentals.get("quick_ratio"), 0.3, 2)
    financial_health = (de_score * 0.5 + cr_score * 0.3 + qr_score * 0.2)

    # ── Growth ──
    rev_growth = _score_between(fundamentals.get("revenue_growth"), -10, 40)
    earn_growth = _score_between(fundamentals.get("earnings_growth"), -20, 50)
    growth = (rev_growth * 0.5 + earn_growth * 0.5)

    # ── Dividend ──
    div_yield = _score_between(fundamentals.get("dividend_yield"), 0, 5)
    payout    = _score_between(fundamentals.get("payout_ratio"), 0, 80)
    dividend  = (div_yield * 0.6 + payout * 0.4)

    return {
        "valuation":        {"score": round(valuation, 4),        "details": {"P/E": pe_score, "P/B": pb_score, "PEG": peg_score}},
        "profitability":    {"score": round(profitability, 4),    "details": {"ROE": roe_score, "ROA": roa_score, "Margin": margin_score, "EPS": eps_score}},
        "financial_health": {"score": round(financial_health, 4), "details": {"D/E": de_score, "Current": cr_score, "Quick": qr_score}},
        "growth":           {"score": round(growth, 4),           "details": {"Revenue": rev_growth, "Earnings": earn_growth}},
        "dividend":         {"score": round(dividend, 4),         "details": {"Yield": div_yield, "Payout": payout}},
    }


def compute_composite_score(factor_scores: dict) -> float:
    """
    Weighted composite fundamental score (0-1).

    >0.6 = Bullish · 0.4-0.6 = Neutral · <0.4 = Bearish
    """
    total = 0.0
    for factor, weight in FUNDAMENTAL_WEIGHTS.items():
        total += factor_scores.get(factor, {}).get("score", 0.5) * weight
    return round(total, 4)
