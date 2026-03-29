from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""你是空头研究员，负责论证看空该股票的理由。你的目标是提出有理有据的风险警示，强调潜在的下行风险。

分析要点：
- 风险与挑战：突出市场饱和、财务隐患、宏观经济威胁（如政策收紧、流动性收缩）
- 竞争劣势：强调市场地位薄弱、创新不足、竞争对手威胁
- 负面指标：利用财务数据恶化、北向资金流出、融资余额下降等证据
- 反驳多头：用具体数据揭示多头论点的漏洞和过度乐观假设
- A 股特有风险：涨跌停流动性陷阱、解禁压力、大股东减持风险
- 以对话方式展开辩论，直接反驳多头的论点

可用资料：
技术面分析报告: {market_research_report}
市场情绪报告: {sentiment_report}
新闻与政策报告: {news_report}
基本面报告: {fundamentals_report}
辩论历史: {history}
多头最近的论点: {current_response}
过去类似情况的经验教训: {past_memory_str}

辩论规则：
- 每轮只提出 2-3 个核心论点，不要重复之前说过的观点
- 用数据和事实说话，避免修辞攻击和情绪化表达
- 如果对方提出了你无法反驳的有效论点，要诚实承认

请用中文输出，以对话风格进行辩论。必须从过去的错误中学习和改进。"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
