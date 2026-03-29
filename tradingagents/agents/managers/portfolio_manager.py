from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_portfolio_manager(llm, memory, market: str = "us"):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        # CN mode extra reports
        capital_flow = state.get("capital_flow_report", "")
        market_sentiment = state.get("market_sentiment_report", "")
        policy = state.get("policy_report", "")
        sector_theme = state.get("sector_theme_report", "")

        extra = ""
        if capital_flow:
            extra += f"\n\n{capital_flow}"
        if market_sentiment:
            extra += f"\n\n{market_sentiment}"
        if policy:
            extra += f"\n\n{policy}"
        if sector_theme:
            extra += f"\n\n{sector_theme}"

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}{extra}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作为投资组合经理，综合风控分析师的辩论结果，做出最终交易决策。

{instrument_context}

---

**评级标准**（选择其一）：
- **买入**: 强烈看好，建议建仓或加仓
- **增持**: 前景向好，建议逐步增加仓位
- **持有**: 维持现有仓位，暂不操作
- **减持**: 建议降低仓位，部分止盈
- **卖出**: 建议清仓或不建仓

**背景信息：**
- 交易员的投资计划: **{trader_plan}**
- 过去决策的经验教训: **{past_memory_str}**

**输出格式要求：**
1. **评级**: 明确给出 买入/增持/持有/减持/卖出 之一
2. **执行摘要**: 简明的行动计划，包括入场策略、仓位比例、关键风控水平和持有周期
3. **投资论点**: 基于分析师辩论和过去反思的详细推理

---

**风控分析师辩论记录：**
{history}

---

做出果断决策，每个结论都必须有分析师辩论中的具体证据支撑。所有输出使用中文。

在报告最后添加免责声明：
> **免责声明：本报告由 AI 自动生成，仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。**"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
