"""Map common company names to Yahoo Finance symbols for yfinance."""

from __future__ import annotations

_ALIASES: dict[str, str] = {
    "TESLA": "TSLA",
    "APPLE": "AAPL",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "FACEBOOK": "META",
    "META PLATFORMS": "META",
    "NETFLIX": "NFLX",
    "NVIDIA": "NVDA",
}


def normalize_equity_ticker(raw: str) -> str:
    s = raw.strip().upper()
    return _ALIASES.get(s, s)
