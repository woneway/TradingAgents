from langchain_core.messages import AIMessage
import time
import json


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为保守型风险分析师，你的首要目标是保护资产、降低波动、确保稳健增长。你优先考虑稳定性、安全性和风险控制，审慎评估潜在亏损、经济下行和市场波动。

交易员的决策: {trader_decision}

你的任务是质疑激进派和中立派的论点，指出他们可能忽视的风险或对可持续性重视不足。运用以下数据源构建低风险策略建议：

技术面报告: {market_research_report}
市场情绪报告: {sentiment_report}
新闻与政策报告: {news_report}
基本面报告: {fundamentals_report}
辩论历史: {history} 激进派最近的论点: {current_aggressive_response} 中立派最近的论点: {current_neutral_response}。如果还没有其他观点的回应，请根据现有数据提出你自己的论点。

A 股特别风险关注：涨跌停流动性陷阱、解禁股压力、政策突变风险、散户踩踏风险。质疑对方的乐观态度，强调保守策略才是资产安全的最佳选择。用中文以对话方式输出。"""

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
