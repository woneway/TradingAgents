from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_news
from tradingagents.agents.utils.astock_tools import get_northbound_flow, get_limit_updown
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_northbound_flow,
            get_limit_updown,
        ]

        system_message = (
            "你是一位 A 股市场情绪分析师，负责分析市场情绪、社交媒体舆论和资金流向。你需要使用工具获取以下数据并撰写综合报告："
            "\n1. 使用 get_news 搜索个股相关的新闻和社交媒体讨论"
            "\n2. 使用 get_northbound_flow 获取北向资金（外资）流向数据"
            "\n3. 使用 get_limit_updown 获取涨跌停统计，了解市场情绪极端状态"
            "\n\n分析重点："
            "\n- 北向资金连续流入/流出的趋势和力度（外资态度）"
            "\n- 涨跌停家数比例（市场情绪温度计）"
            "\n- 社交媒体和新闻中的舆论倾向"
            "\n- 散户情绪指标（换手率异常、跟风盘等）"
            "\n\n提供具体、可操作的情绪面判断，帮助交易者理解当前市场氛围。所有分析和输出必须使用中文。"
            + """ 请在报告末尾附上Markdown表格，整理报告中的关键要点。"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个 AI 分析助手，与其他助手协作完成分析任务。"
                    " 使用提供的工具推进分析。如果无法完全回答，其他助手会接续你的工作。"
                    " 如果你或其他助手已经得出最终交易建议，请在回复前加上 FINAL TRANSACTION PROPOSAL: **买入/持有/卖出**。"
                    " 可用工具: {tool_names}。\n{system_message}"
                    "当前日期: {current_date}。{instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
