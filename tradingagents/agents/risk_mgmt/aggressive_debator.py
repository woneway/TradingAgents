import time
import json


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

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

        prompt = f"""作为激进型风险分析师，你的角色是积极主张高风险高回报的策略，强调大胆的投资机会和竞争优势。评估交易员的决策时，专注于潜在的上行空间、增长潜力和创新收益。

交易员的决策: {trader_decision}

你的任务是为交易员的决策构建有说服力的看多论证，质疑保守派和中立派的过度谨慎。运用以下数据源支持你的论点：

技术面报告: {market_research_report}
市场情绪报告: {sentiment_report}
新闻与政策报告: {news_report}
基本面报告: {fundamentals_report}{extra}
辩论历史: {history} 保守派最近的论点: {current_conservative_response} 中立派最近的论点: {current_neutral_response}。如果还没有其他观点的回应，请根据现有数据提出你自己的论点。

积极回应对方的具体关切，反驳其逻辑漏洞，主张大胆出击才能跑赢市场。聚焦辩论和说服，而非简单堆砌数据。用中文以对话方式输出。"""

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
