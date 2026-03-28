import functools
import time
import json

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"基于分析团队的综合研究，以下是为 {company_name} 制定的投资计划。{instrument_context} 该计划整合了技术面趋势、宏观经济指标、政策面和市场情绪分析。请以此为基础做出交易决策。\n\n投资计划: {investment_plan}\n\n请利用这些信息做出明智的战略决策。",
        }

        messages = [
            {
                "role": "system",
                "content": f"""你是一位 A 股交易员，基于分析团队的研究做出投资决策。根据分析结果，给出明确的买入、卖出或持有建议。必须以 'FINAL TRANSACTION PROPOSAL: **买入/持有/卖出**' 结尾确认你的建议。运用过去的交易经验改进你的分析。以下是类似情况的历史反思和经验教训: {past_memory_str}""",
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
