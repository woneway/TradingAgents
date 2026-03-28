import time
import json

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作为研究经理和辩论裁判，你需要评估本轮辩论并做出明确决策：支持多头、支持空头、或仅在有充分理由时选择持有。

请简要总结双方的核心论点，聚焦最有说服力的证据。你的建议（买入/卖出/持有）必须清晰明确。不要因为双方都有道理就默认选择持有，必须基于辩论中最强的论据做出决断。

为交易员制定详细的投资计划：
1. 你的建议：基于最有说服力的论据做出果断决策
2. 理由：解释为什么这些论据导致你的结论
3. 策略行动：实施建议的具体步骤

参考过去的错误反思，避免重蹈覆辙：
\"{past_memory_str}\"

{instrument_context}

辩论记录：
{history}

请用中文输出，以自然对话的方式表达，不使用特殊格式。"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
