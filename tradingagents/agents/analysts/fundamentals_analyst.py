from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
)
from tradingagents.agents.utils.astock_tools import (
    get_margin_data,
    get_share_unlock,
    get_st_status,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_margin_data,
            get_share_unlock,
            get_st_status,
        ]

        system_message = (
            "你是一位 A 股基本面分析师，负责分析公司的财务状况和基本面数据。"
            "\n\n使用工具获取以下数据："
            "\n- get_fundamentals: 综合基本面数据（PE、PB、市值、换手率等）"
            "\n- get_balance_sheet: 资产负债表"
            "\n- get_cashflow: 现金流量表"
            "\n- get_income_statement: 利润表"
            "\n- get_margin_data: 融资融券数据（融资余额变化反映杠杆资金态度）"
            "\n- get_share_unlock: 限售股解禁日历（未来 90 天解禁计划，评估抛压风险）"
            "\n- get_st_status: ST 状态查询（是否有退市风险）"
            "\n\n分析重点："
            "\n1. 估值水平：PE/PB 与行业平均和历史分位的对比"
            "\n2. 盈利能力：营收增长率、净利润增长率、ROE 趋势"
            "\n3. 财务健康：资产负债率、现金流充裕度、有息负债率"
            "\n4. 融资融券：融资余额变化趋势（增加=看多杠杆增加）"
            "\n5. 特别关注：商誉减值风险、应收账款异常、关联交易"
            "\n\n提供具体、可操作的基本面判断。所有分析和输出必须使用中文。"
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
