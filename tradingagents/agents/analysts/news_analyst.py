from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_news,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "你是一位 A 股新闻与政策分析师，负责分析近期新闻、宏观政策和行业动态对个股的影响。"
            "\n\n使用 get_news 搜索个股相关新闻，使用 get_global_news 获取宏观经济和政策新闻。"
            "\n\n分析重点："
            "\n1. 证监会、央行等监管机构的最新政策动向"
            "\n2. 宏观经济数据（GDP、CPI、PMI、社融等）对市场的影响"
            "\n3. 行业政策变化（产业扶持、监管收紧等）"
            "\n4. 公司层面的重大事件（并购、定增、业绩预告等）"
            "\n5. 国际因素（中美关系、汇率变动、外围市场等）"
            "\n\nA 股是典型的「政策市」，政策面分析权重应高于其他市场。"
            "\n提供具体、可操作的新闻面判断。所有分析和输出必须使用中文。"
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
            "news_report": report,
        }

    return news_analyst_node
