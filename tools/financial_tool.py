from __future__ import annotations

from typing import Type

import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tools.ticker_resolve import resolve_yahoo_finance_symbol


class FinancialToolInput(BaseModel):
    ticker: str = Field(
        description=(
            "Ticker: US (AAPL), or Indian NSE/BSE (RELIANCE.NS, TCS.NS). "
            "Plain Indian symbols are auto-resolved when possible."
        )
    )


class StockFinancialTool(BaseTool):
    name: str = "Stock Financial Data Fetcher"
    description: str = (
        "Fetches comprehensive financial data for a given stock ticker. "
        "Returns current price, valuation ratios, income statement, balance sheet, "
        "and analyst recommendations."
    )
    args_schema: Type[BaseModel] = FinancialToolInput
    # Caps redundant LLM tool loops (CrewAI default agent max_iter is 25).
    max_usage_count: int | None = 2

    def _run(self, ticker: str) -> str:
        ticker = resolve_yahoo_finance_symbol(ticker)
        try:
            t = yf.Ticker(ticker)
            sections = [
                self._price_section(t, ticker),
                self._valuation_section(t),
                self._income_section(t),
                self._balance_section(t),
                self._recommendations_section(t),
            ]
            return "\n\n".join(s for s in sections if s)
        except Exception as e:
            return f"Error fetching financial data for {ticker}: {e}"

    # ------------------------------------------------------------------
    def _price_section(self, t: yf.Ticker, ticker: str) -> str:
        info = t.info or {}
        cur = info.get("currency") or "USD"
        lines = [f"=== Price Data: {ticker} ==="]
        lines.append(f"Exchange        : {info.get('exchange', 'N/A')}")
        lines.append(f"Currency        : {cur}")
        lines.append(f"Current Price   : {_v(info, 'currentPrice')}")
        lines.append(f"Previous Close  : {_v(info, 'previousClose')}")
        lines.append(f"52-Week High    : {_v(info, 'fiftyTwoWeekHigh')}")
        lines.append(f"52-Week Low     : {_v(info, 'fiftyTwoWeekLow')}")
        lines.append(f"Market Cap      : {_fmt_large(info.get('marketCap'), cur)}")
        lines.append(f"Sector          : {info.get('sector', 'N/A')}")
        lines.append(f"Industry        : {info.get('industry', 'N/A')}")
        return "\n".join(lines)

    def _valuation_section(self, t: yf.Ticker) -> str:
        info = t.info or {}
        lines = ["=== Valuation ==="]
        lines.append(f"Trailing P/E    : {_v(info, 'trailingPE')}")
        lines.append(f"Forward P/E     : {_v(info, 'forwardPE')}")
        lines.append(f"Trailing EPS    : {_v(info, 'trailingEps')}")
        lines.append(f"Dividend Yield  : {_pct(info.get('dividendYield'))}")
        lines.append(f"Beta            : {_v(info, 'beta')}")
        lines.append(f"Price/Book      : {_v(info, 'priceToBook')}")
        return "\n".join(lines)

    def _income_section(self, t: yf.Ticker) -> str:
        try:
            df = t.income_stmt
            if df is None or df.empty:
                return "=== Income Statement ===\nData unavailable."
            cols = df.columns[:2]
            lines = [f"=== Income Statement (last 2 years) ==="]
            lines.append(f"{'Metric':<30} " + "  ".join(str(c)[:10] for c in cols))
            cur = (t.info or {}).get("currency") or "USD"
            for row in ["Total Revenue", "Gross Profit", "Net Income", "EBITDA"]:
                if row in df.index:
                    vals = "  ".join(_fmt_large(df.loc[row, c], cur) for c in cols)
                    lines.append(f"{row:<30} {vals}")
            return "\n".join(lines)
        except Exception:
            return "=== Income Statement ===\nData unavailable."

    def _balance_section(self, t: yf.Ticker) -> str:
        try:
            df = t.balance_sheet
            if df is None or df.empty:
                return "=== Balance Sheet ===\nData unavailable."
            col = df.columns[0]
            lines = [f"=== Balance Sheet (latest: {str(col)[:10]}) ==="]
            cur = (t.info or {}).get("currency") or "USD"
            for row in [
                "Total Assets",
                "Total Liabilities Net Minority Interest",
                "Total Debt",
                "Cash And Cash Equivalents",
                "Stockholders Equity",
            ]:
                if row in df.index:
                    lines.append(f"{row:<40}: {_fmt_large(df.loc[row, col], cur)}")
            return "\n".join(lines)
        except Exception:
            return "=== Balance Sheet ===\nData unavailable."

    def _recommendations_section(self, t: yf.Ticker) -> str:
        try:
            df = t.recommendations
            if df is None or df.empty:
                return "=== Analyst Recommendations ===\nData unavailable."
            recent = df.tail(5)
            lines = ["=== Analyst Recommendations (recent 5) ==="]
            for _, row in recent.iterrows():
                firm = row.get("Firm", row.get("firm", "Unknown"))
                to_grade = row.get("To Grade", row.get("toGrade", ""))
                action = row.get("Action", row.get("action", ""))
                lines.append(f"  {firm}: {to_grade}  [{action}]")
            return "\n".join(lines)
        except Exception:
            return "=== Analyst Recommendations ===\nData unavailable."


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _v(d: dict, key: str) -> str:
    val = d.get(key)
    return "N/A" if val is None else str(round(val, 2))


def _pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{round(val * 100, 2)}%"


def _money_prefix(currency: str | None) -> str:
    c = (currency or "USD").upper()
    if c == "INR":
        return "₹"
    if c == "USD":
        return "$"
    return f"{c} "


def _fmt_large(val, currency: str | None = None) -> str:
    if val is None:
        return "N/A"
    try:
        val = float(val)
        p = _money_prefix(currency)
        if abs(val) >= 1e12:
            return f"{p}{val/1e12:.2f}T"
        if abs(val) >= 1e9:
            return f"{p}{val/1e9:.2f}B"
        if abs(val) >= 1e6:
            return f"{p}{val/1e6:.2f}M"
        if p.endswith(" "):
            return f"{p}{val:,.0f}"
        return f"{p}{val:,.0f}"
    except Exception:
        return str(val)
