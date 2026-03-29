"""
Stock Analysis Agent — entry point.

Usage:
    python main.py                 # default ticker (see code) or pass tickers as args
    python main.py TSLA AAPL       # US tickers (Yahoo: plain symbol)
    python main.py RELIANCE TCS    # Indian NSE: unqualified symbols resolve to .NS / .BO
    python main.py RELIANCE.NS     # or pass Yahoo symbols explicitly (NSE/BSE)

Yahoo Finance / yfinance:
    US: AAPL, MSFT
    India NSE: e.g. RELIANCE.NS, TCS.NS (many work as RELIANCE, TCS via auto-resolve)
    India BSE: e.g. 500325.BO — pass the full Yahoo symbol when needed
    Note: symbols that trade in the US as ADRs (e.g. INFY) resolve to the US listing unless you pass INFY.NS.

Environment (optional):
    STOCK_ANALYSIS_OLLAMA_MODEL — default ollama/llama3.1:latest (must support tools)
    OLLAMA_BASE_URL             — default http://localhost:11434

Requires: `ollama pull llama3.1` (or another tool-capable model you set via env).
"""

from __future__ import annotations

import os
import sys

from crewai import LLM

from crew import build_stock_analysis_crew

# Ollama models must support tool/function calling for agents that use CrewAI tools.
# deepseek-r1:1.5b (and many reasoning-only variants) return 400: "does not support tools".
_DEFAULT_OLLAMA_MODEL = "ollama/llama3.1:latest"


def run(ticker: str, make_crew) -> None:
    print(f"\n{'='*60}")
    print(f"  STOCK ANALYSIS: {ticker.upper()}")
    print(f"{'='*60}\n")

    crew = make_crew(ticker)
    result = crew.kickoff()

    print(f"\n{'='*60}")
    print(f"  FINAL RECOMMENDATION: {ticker.upper()}")
    print(f"{'='*60}")
    print(result)
    print()


def main() -> None:
    model = os.environ.get("STOCK_ANALYSIS_OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = LLM(model=model, base_url=base_url)

    make_crew = build_stock_analysis_crew(llm)

    tickers = [arg.upper() for arg in sys.argv[1:]] if len(sys.argv) > 1 else ["vedanta"]

    for ticker in tickers:
        run(ticker, make_crew)


if __name__ == "__main__":
    main()
