import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # CN mode extra reports
        capital_flow = state.get("capital_flow_report", "")
        market_sentiment = state.get("market_sentiment_report", "")
        policy = state.get("policy_report", "")
        sector_theme = state.get("sector_theme_report", "")

        extra = ""
        if capital_flow:
            extra += f"\n资金流向报告: {capital_flow}"
        if market_sentiment:
            extra += f"\n市场情绪报告: {market_sentiment}"
        if policy:
            extra += f"\n政策分析报告: {policy}"
        if sector_theme:
            extra += f"\n板块题材报告: {sector_theme}"

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为中立型风险分析师，你的角色是提供平衡的视角，权衡交易员决策的收益和风险。你优先采取全面的分析方法，同时考虑上行和下行因素。

交易员的决策: {trader_decision}

你的任务是质疑激进派和保守派，指出他们各自的偏颇之处。运用以下数据源提出温和可持续的策略建议：

技术面报告: {market_research_report}
市场情绪报告: {sentiment_report}
新闻与政策报告: {news_report}
基本面报告: {fundamentals_report}{extra}
辩论历史: {history} 激进派最近的论点: {current_aggressive_response} 保守派最近的论点: {current_conservative_response}。如果还没有其他观点的回应，请根据现有数据提出你自己的论点。

批判性地分析双方论点，展示为什么平衡的风险策略能在获取增长的同时防范极端波动。聚焦辩论而非简单陈述数据。用中文以对话方式输出。"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
