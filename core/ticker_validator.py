"""
Ticker validation for Indian equities (NSE / BSE).

Flow:
  1. User enters a bare symbol (e.g. "RELIANCE").
  2. We try `{symbol}.NS` (NSE) first — this is the dominant exchange.
  3. If that fails we try `{symbol}.BO` (BSE).
  4. We verify the result is an equity (not ETF / MF / index).
  5. Return a clean info dict for the confirmation prompt.
"""

import yfinance as yf


def validate_ticker(symbol: str) -> dict | None:
    """
    Validate an Indian equity ticker symbol.

    Parameters
    ----------
    symbol : str
        Bare ticker like "RELIANCE", "TCS", "INFY".
        Also accepts pre-suffixed forms like "RELIANCE.NS".

    Returns
    -------
    dict | None
        Company info dict on success, ``None`` if invalid / not found.

        Keys:
          - symbol       : str  — full Yahoo symbol ("RELIANCE.NS")
          - short_name    : str  — company short name
          - long_name     : str  — full registered name
          - sector        : str
          - industry      : str
          - exchange      : str  — "NSE" or "BSE"
          - market_cap    : float | None
          - currency      : str
          - quote_type    : str  — "EQUITY", etc.
    """
    symbol = symbol.strip().upper()

    # If user already appended .NS or .BO, honour it
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return _try_symbol(symbol)

    # Try NSE first, then BSE
    result = _try_symbol(f"{symbol}.NS")
    if result is not None:
        return result

    result = _try_symbol(f"{symbol}.BO")
    return result


def _try_symbol(full_symbol: str) -> dict | None:
    """Attempt to fetch info for *full_symbol* and verify it is an Indian equity."""
    try:
        ticker = yf.Ticker(full_symbol)
        info = ticker.info

        # yfinance returns an almost-empty dict for invalid tickers
        if not info or info.get("regularMarketPrice") is None:
            return None

        # Reject non-equity types (ETFs, mutual funds, indices)
        quote_type = info.get("quoteType", "").upper()
        if quote_type not in ("EQUITY", ""):
            return None

        exchange_suffix = full_symbol.rsplit(".", 1)[-1]
        exchange = "NSE" if exchange_suffix == "NS" else "BSE"

        return {
            "symbol":      full_symbol,
            "short_name":  info.get("shortName", "N/A"),
            "long_name":   info.get("longName", info.get("shortName", "N/A")),
            "sector":      info.get("sector", "N/A"),
            "industry":    info.get("industry", "N/A"),
            "exchange":    exchange,
            "market_cap":  info.get("marketCap"),
            "currency":    info.get("currency", "INR"),
            "quote_type":  quote_type or "EQUITY",
        }
    except Exception:
        return None


def format_market_cap(value) -> str:
    """Human-readable market cap string in Indian format (Cr / Lakh Cr)."""
    if value is None:
        return "N/A"
    crore = value / 1e7
    if crore >= 1e5:
        return f"₹{crore / 1e5:,.2f} Lakh Cr"
    elif crore >= 1:
        return f"₹{crore:,.0f} Cr"
    else:
        return f"₹{value:,.0f}"
