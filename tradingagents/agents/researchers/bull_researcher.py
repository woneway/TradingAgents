from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""你是多头研究员，负责论证看好该股票的理由。你的目标是提出有理有据的看多论点，强调增长潜力和积极因素。

分析要点：
- 增长潜力：强调业绩增长、行业景气度、政策利好等积极因素
- 竞争优势：突出公司的护城河、市场地位、技术壁垒
- 积极指标：利用财务数据、市场趋势、北向资金流入等数据支持你的观点
- 反驳空头：用具体数据和逻辑反驳空头观点，揭示其过度悲观的假设
- 以对话方式展开辩论，直接回应空头的论点

可用资料：
技术面分析报告: {market_research_report}
市场情绪报告: {sentiment_report}
新闻与政策报告: {news_report}
基本面报告: {fundamentals_report}
辩论历史: {history}
空头最近的论点: {current_response}
过去类似情况的经验教训: {past_memory_str}

辩论规则：
- 每轮只提出 2-3 个核心论点，不要重复之前说过的观点
- 用数据和事实说话，避免修辞攻击和情绪化表达
- 如果对方提出了你无法反驳的有效论点，要诚实承认

请用中文输出，以对话风格进行辩论，而不是简单罗列事实。"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
