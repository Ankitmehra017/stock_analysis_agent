from __future__ import annotations

from typing import Type

import feedparser
import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tools.ticker_resolve import resolve_yahoo_finance_symbol


class NewsToolInput(BaseModel):
    ticker: str = Field(
        description=(
            "Stock ticker: US (e.g. AAPL), or Indian NSE/BSE as RELIANCE.NS, TCS.NS "
            "(unqualified Indian names like RELIANCE are resolved automatically when possible)."
        )
    )


class StockNewsTool(BaseTool):
    name: str = "Stock News Fetcher"
    description: str = (
        "Fetches recent news headlines and summaries for a given stock ticker. "
        "Returns a formatted list of up to 10 recent articles with title, source, date, and URL."
    )
    args_schema: Type[BaseModel] = NewsToolInput
    # Caps redundant LLM tool loops (CrewAI default agent max_iter is 25).
    max_usage_count: int | None = 2

    def _run(self, ticker: str) -> str:
        ticker = resolve_yahoo_finance_symbol(ticker)
        articles = self._fetch_yfinance_news(ticker)
        if not articles:
            articles = self._fetch_rss_news(ticker)
        if not articles:
            return f"No recent news found for {ticker}."
        return self._format_articles(ticker, articles)

    # ------------------------------------------------------------------
    # Primary: yfinance ticker.news
    # ------------------------------------------------------------------
    def _fetch_yfinance_news(self, ticker: str) -> list[dict]:
        try:
            t = yf.Ticker(ticker)
            raw = t.news or []
            results = []
            for item in raw[:10]:
                content = item.get("content", {})
                results.append({
                    "title": content.get("title", item.get("title", "No title")),
                    "publisher": content.get("provider", {}).get("displayName", item.get("publisher", "Unknown")),
                    "url": content.get("canonicalUrl", {}).get("url", item.get("link", "")),
                    "date": _format_ts(content.get("pubDate") or item.get("providerPublishTime")),
                    "summary": content.get("summary", ""),
                })
            return results
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Fallback: Yahoo Finance RSS
    # ------------------------------------------------------------------
    def _fetch_rss_news(self, ticker: str) -> list[dict]:
        try:
            region, lang = (
                ("IN", "en-IN")
                if ticker.endswith((".NS", ".BO"))
                else ("US", "en-US")
            )
            url = (
                f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                f"?s={ticker}&region={region}&lang={lang}"
            )
            feed = feedparser.parse(url)
            results = []
            for entry in feed.entries[:10]:
                results.append({
                    "title": entry.get("title", "No title"),
                    "publisher": entry.get("source", {}).get("title", "Yahoo Finance"),
                    "url": entry.get("link", ""),
                    "date": entry.get("published", ""),
                    "summary": entry.get("summary", ""),
                })
            return results
        except Exception:
            return []

    # ------------------------------------------------------------------
    def _format_articles(self, ticker: str, articles: list[dict]) -> str:
        lines = [f"=== Recent News for {ticker} ===\n"]
        for i, a in enumerate(articles, 1):
            lines.append(f"{i}. {a['title']}")
            lines.append(f"   Source : {a['publisher']}  |  Date: {a['date']}")
            if a.get("url"):
                lines.append(f"   URL    : {a['url']}")
            if a.get("summary"):
                lines.append(f"   Summary: {a['summary'][:200]}")
            lines.append("")
        return "\n".join(lines)


def _format_ts(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        try:
            import datetime
            return datetime.datetime.utcfromtimestamp(value).strftime("%Y-%m-%d")
        except Exception:
            return str(value)
    return str(value)
