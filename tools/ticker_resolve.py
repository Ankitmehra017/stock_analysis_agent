"""
Resolve user input to a Yahoo Finance symbol yfinance can quote.

US: plain tickers (AAPL). India: NSE uses .NS, BSE uses .BO (e.g. RELIANCE.NS).
Unqualified symbols try US first, then NSE, then BSE so Indian names work without suffix.
"""

from __future__ import annotations

import yfinance as yf

from tools.ticker_aliases import normalize_equity_ticker


def _has_tradable_data(ticker: yf.Ticker) -> bool:
    """True if Yahoo returns usable price/quote data for this symbol."""
    try:
        fi = ticker.fast_info
        if fi is not None:
            # yfinance FastInfo is dict-like
            fd = dict(fi) if hasattr(fi, "keys") else {}
            for key in ("last_price", "lastPrice", "open", "previous_close"):
                v = fd.get(key) if isinstance(fd, dict) else getattr(fi, key, None)
                if v is not None and isinstance(v, (int, float)):
                    return True
    except Exception:
        pass
    try:
        info = ticker.info or {}
        for key in ("regularMarketPrice", "currentPrice", "previousClose"):
            v = info.get(key)
            if v is not None and isinstance(v, (int, float)):
                return True
        if info.get("quoteType") == "EQUITY" and info.get("marketCap"):
            return True
    except Exception:
        pass
    return False


def resolve_yahoo_finance_symbol(raw: str) -> str:
    """
    Return the best Yahoo Finance symbol for yfinance (e.g. RELIANCE -> RELIANCE.NS).
    Qualified symbols (contain '.') are used if they quote; otherwise NSE/BSE fallbacks are tried.
    """
    s = normalize_equity_ticker(raw.strip())
    if not s:
        return s

    if "." in s:
        t = yf.Ticker(s)
        if _has_tradable_data(t):
            return s
        base = s.rsplit(".", 1)[0]
        for sym in (f"{base}.NS", f"{base}.BO"):
            if sym == s:
                continue
            t2 = yf.Ticker(sym)
            if _has_tradable_data(t2):
                return sym
        return s

    for sym in (s, f"{s}.NS", f"{s}.BO"):
        t = yf.Ticker(sym)
        if _has_tradable_data(t):
            return sym
    return s
