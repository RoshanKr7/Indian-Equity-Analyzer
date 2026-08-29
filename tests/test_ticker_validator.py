"""Tests for ticker validation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_validate_valid_nse():
    """RELIANCE should resolve on NSE."""
    from core.ticker_validator import validate_ticker
    result = validate_ticker("RELIANCE")
    assert result is not None
    assert result["exchange"] == "NSE"
    assert "RELIANCE" in result["symbol"]
    assert result["quote_type"] == "EQUITY"


def test_validate_valid_with_suffix():
    """Pre-suffixed ticker should work."""
    from core.ticker_validator import validate_ticker
    result = validate_ticker("TCS.NS")
    assert result is not None
    assert result["symbol"] == "TCS.NS"


def test_validate_invalid_ticker():
    """Non-existent ticker should return None."""
    from core.ticker_validator import validate_ticker
    result = validate_ticker("XYZABCDEF123")
    assert result is None


def test_format_market_cap():
    """Market cap formatting in Indian style."""
    from core.ticker_validator import format_market_cap
    assert "Lakh Cr" in format_market_cap(15_00_000_00_00_000)  # 15 lakh crore
    assert "Cr" in format_market_cap(50_000_00_00_000)  # 50k crore
    assert format_market_cap(None) == "N/A"


if __name__ == "__main__":
    test_validate_valid_nse()
    print("✅ test_validate_valid_nse passed")
    test_format_market_cap()
    print("✅ test_format_market_cap passed")
    print("All basic tests passed!")
