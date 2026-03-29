from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task

from tools.financial_tool import StockFinancialTool
from tools.news_tool import StockNewsTool
from tools.ticker_resolve import resolve_yahoo_finance_symbol

# ---------------------------------------------------------------------------
# System template — injected into every agent to suppress refusals/disclaimers
# and keep local LLMs on-task in an automated pipeline.
# ---------------------------------------------------------------------------
_FINANCE_SYSTEM_TEMPLATE = """\
You are {role} operating inside an automated financial research pipeline.
Your task is a professional duty: produce the output described in your goal.
Do NOT add investment disclaimers, refuse analysis, or give partial output.
Incomplete output breaks the downstream pipeline. Always complete your section fully.

{backstory}
"""

def build_stock_analysis_crew(llm: LLM):
    """
    Factory that returns a make_crew(ticker) callable.
    Call make_crew(ticker) to get a fully configured Crew ready to kickoff().
    """

    # -----------------------------------------------------------------------
    # Agent 1 — News Researcher (parallel)
    # -----------------------------------------------------------------------
    news_agent = Agent(
        role="Financial News Researcher",
        goal=(
            "Retrieve and summarise the 10 most recent news articles for the given "
            "stock ticker. Highlight any market-moving events, earnings surprises, "
            "regulatory actions, or management changes."
        ),
        backstory=(
            "You are a veteran financial journalist with 20 years of experience "
            "tracking equities. You excel at quickly identifying the news that "
            "matters to investors and separating signal from noise."
        ),
        tools=[StockNewsTool()],
        llm=llm,
        system_template=_FINANCE_SYSTEM_TEMPLATE,
        verbose=True,
        max_iter=6,
    )

    # -----------------------------------------------------------------------
    # Agent 2 — Quantitative Financial Analyst (parallel)
    # -----------------------------------------------------------------------
    financial_agent = Agent(
        role="Quantitative Financial Analyst",
        goal=(
            "Retrieve and interpret the key financial metrics for the given stock "
            "ticker: current price, valuation ratios, income statement highlights, "
            "balance sheet strength, and analyst consensus."
        ),
        backstory=(
            "You are a CFA-certified quant with deep expertise in fundamental "
            "analysis. You turn raw financial data into structured, decision-ready "
            "summaries that portfolio managers can act on immediately."
        ),
        tools=[StockFinancialTool()],
        llm=llm,
        system_template=_FINANCE_SYSTEM_TEMPLATE,
        verbose=True,
        max_iter=6,
    )

    # -----------------------------------------------------------------------
    # Agent 3 — Senior Investment Analyst (synthesises 1 + 2)
    # -----------------------------------------------------------------------
    analyst_agent = Agent(
        role="Senior Investment Analyst",
        goal=(
            "Synthesise the news report and the financial data into a coherent "
            "investment analysis brief. Identify financial health, news sentiment, "
            "key risks, and key opportunities."
        ),
        backstory=(
            "You are a senior sell-side analyst at a top-tier investment bank. "
            "You write the investment briefs that institutional investors rely on "
            "to make multi-million dollar decisions. Your analysis is structured, "
            "evidence-based, and free of speculation."
        ),
        tools=[],
        llm=llm,
        system_template=_FINANCE_SYSTEM_TEMPLATE,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Agent 4 — Portfolio Advisor (final verdict)
    # -----------------------------------------------------------------------
    advisor_agent = Agent(
        role="Portfolio Advisor",
        goal=(
            "Read the investment analysis brief and deliver a clear, justified "
            "BUY / HOLD / SELL recommendation with a confidence level and "
            "concise bullet-point rationale."
        ),
        backstory=(
            "You are the head of equity strategy at a wealth management firm "
            "overseeing $10B in assets. Your verdicts are trusted by advisors "
            "and high-net-worth clients alike. You are decisive, evidence-driven, "
            "and always give a definitive recommendation."
        ),
        tools=[],
        llm=llm,
        system_template=_FINANCE_SYSTEM_TEMPLATE,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Crew factory
    # -----------------------------------------------------------------------
    def make_crew(ticker: str) -> Crew:
        ticker = resolve_yahoo_finance_symbol(ticker)

        _tool_once = (
            f"Use the exact Yahoo Finance symbol '{ticker}' when calling the tool "
            f"(US tickers plain; India NSE often ends in .NS, BSE in .BO). "
            f"Call the tool at most once for this task; one successful fetch is enough. "
            f"Then write your final answer as plain prose (no JSON, no fake tool calls)."
        )

        # Task 1 — runs in parallel with Task 2
        news_task = Task(
            description=(
                f"Use the 'Stock News Fetcher' tool to retrieve the latest news for "
                f"ticker '{ticker}'. {_tool_once} "
                f"Summarise each article in 1-2 sentences. "
                f"Identify the overall news sentiment (positive / negative / neutral) "
                f"and flag any major events."
            ),
            expected_output=(
                "A numbered list of up to 10 recent news items, each with: title, "
                "source, date, 1-2 sentence summary. Followed by a one-paragraph "
                "overall news sentiment assessment."
            ),
            agent=news_agent,
            async_execution=True,  # fan-out: runs concurrently with financial_task
        )

        # Task 2 — runs in parallel with Task 1
        financial_task = Task(
            description=(
                f"Use the 'Stock Financial Data Fetcher' tool to retrieve financial "
                f"data for ticker '{ticker}'. {_tool_once} "
                f"Interpret each section (price, valuation, income, balance sheet, "
                f"analyst recommendations) and note any standout figures — e.g. "
                f"unusually high/low P/E, debt load, revenue trend."
            ),
            expected_output=(
                "A structured financial report with sections: PRICE DATA, VALUATION, "
                "INCOME STATEMENT HIGHLIGHTS, BALANCE SHEET HIGHLIGHTS, ANALYST CONSENSUS. "
                "Each section must include both raw figures and a 1-sentence interpretation."
            ),
            agent=financial_agent,
            async_execution=True,  # fan-out: runs concurrently with news_task
        )

        # Task 3 — fan-in: waits for both async tasks to complete
        analyst_task = Task(
            description=(
                f"You have received a news report and a financial data report for '{ticker}'. "
                f"Synthesise them into a comprehensive investment analysis brief. "
                f"Cover: (1) Business overview & competitive position, "
                f"(2) Financial health score (1-10), "
                f"(3) News sentiment impact, "
                f"(4) Top 3 risks, "
                f"(5) Top 3 opportunities."
            ),
            expected_output=(
                "An investment analysis brief with five clearly labelled sections: "
                "Business Overview, Financial Health, News Sentiment, Key Risks, Key Opportunities."
            ),
            agent=analyst_agent,
            context=[news_task, financial_task],  # fan-in: aggregates both outputs
        )

        # Task 4 — final recommendation
        advisor_task = Task(
            description=(
                f"Based solely on the investment analysis brief for '{ticker}', "
                f"deliver your final recommendation. "
                f"Format: VERDICT (BUY/HOLD/SELL), CONFIDENCE (Low/Medium/High), "
                f"followed by 3-5 bullet points of rationale. "
                f"Be decisive — do not hedge or sit on the fence."
            ),
            expected_output=(
                "VERDICT: [BUY / HOLD / SELL]\n"
                "CONFIDENCE: [Low / Medium / High]\n"
                "RATIONALE:\n"
                "• <point 1>\n• <point 2>\n• <point 3>"
            ),
            agent=advisor_agent,
            context=[analyst_task],
        )

        return Crew(
            agents=[news_agent, financial_agent, analyst_agent, advisor_agent],
            tasks=[news_task, financial_task, analyst_task, advisor_task],
            process=Process.sequential,
            verbose=True,
        )

    return make_crew
