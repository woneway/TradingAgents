import functools
import time
import json

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_trader(llm, memory, market: str = "us"):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # CN mode: include additional reports if available
        capital_flow = state.get("capital_flow_report", "")
        market_sentiment = state.get("market_sentiment_report", "")
        policy = state.get("policy_report", "")
        sector_theme = state.get("sector_theme_report", "")
        is_cn = market == "cn"

        extra_reports = ""
        if capital_flow:
            extra_reports += f"\n\n资金流向报告: {capital_flow}"
        if market_sentiment:
            extra_reports += f"\n\n市场情绪报告: {market_sentiment}"
        if policy:
            extra_reports += f"\n\n政策分析报告: {policy}"
        if sector_theme:
            extra_reports += f"\n\n板块题材报告: {sector_theme}"

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}{extra_reports}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        astock_rules = ""
        if is_cn:
            astock_rules = (
                "\n\nA 股交易规则（必须遵守）："
                "\n1. T+1 制度：今日买入明日才能卖出。不能日内止损，买入时必须考虑次日风险。"
                "\n2. 涨跌停限制：主板 ±10%，创业板/科创板 ±20%，ST 股 ±5%。"
                "\n3. 仓位管理：单只不超总资金 30%，首次建仓 1/3 仓位，确认后加仓。"
                "\n4. 输出需包含：建议仓位比例、买入时段、T+1 风险提示。"
            )

        context = {
            "role": "user",
            "content": f"基于分析团队的综合研究，以下是为 {company_name} 制定的投资计划。{instrument_context} 该计划整合了技术面趋势、宏观经济指标、政策面和市场情绪分析。请以此为基础做出交易决策。\n\n投资计划: {investment_plan}\n\n请利用这些信息做出明智的战略决策。",
        }

        messages = [
            {
                "role": "system",
                "content": f"""你是一位 A 股交易员，基于分析团队的研究做出投资决策。根据分析结果，给出明确的买入、卖出或持有建议。必须以 'FINAL TRANSACTION PROPOSAL: **买入/持有/卖出**' 结尾确认你的建议。运用过去的交易经验改进你的分析。以下是类似情况的历史反思和经验教训: {past_memory_str}{astock_rules}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
