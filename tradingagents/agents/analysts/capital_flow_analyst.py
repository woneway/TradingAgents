from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.astock_tools import (
    get_northbound_flow,
    get_margin_data,
    get_block_trade,
    get_dragon_tiger,
)


def create_capital_flow_analyst(llm):
    def capital_flow_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_northbound_flow,
            get_margin_data,
            get_block_trade,
            get_dragon_tiger,
        ]

        system_message = (
            "你是 A 股资金流向分析师。你的核心任务是通过资金数据判断主力资金的态度和方向。"
            "\n\n请依次调用以下工具获取数据："
            "\n1. get_northbound_flow — 北向资金（外资风向标）"
            "\n2. get_margin_data — 融资融券（内资杠杆方向）"
            "\n3. get_block_trade — 大宗交易（机构暗盘态度）"
            "\n4. get_dragon_tiger — 龙虎榜（异动日主力动向）"
            "\n\n分析框架："
            "\n1. 北向资金（外资风向标）"
            "\n   - 近 10 日净流入/流出趋势"
            "\n   - 单日大额异动（>20亿）的信号意义"
            "\n   - 是否与大盘走势背离（背离往往是反转信号）"
            "\n2. 融资融券（内资杠杆方向）"
            "\n   - 融资余额变化趋势 = 多头力量"
            "\n   - 融券余额变化趋势 = 空头力量"
            "\n   - 融资/融券比值变化"
            "\n3. 大宗交易（机构暗盘态度）"
            "\n   - 折价率：折价越大，卖方越急"
            "\n   - 溢价成交：机构看好，愿意高于市价买入"
            "\n   - 买方营业部是否为知名机构席位"
            "\n4. 龙虎榜（异动日主力动向）"
            "\n   - 机构席位 vs 游资席位"
            "\n   - 净买入还是净卖出"
            "\n   - 买卖集中度"
            "\n\n输出要求："
            "\n- 明确给出资金面的多空判断（资金净流入/净流出/分歧）"
            "\n- 标注最关键的 1-2 个资金信号"
            "\n- 如果某项数据为空，说明这是正常现象，不要强行解读"
            "\n- 所有分析和输出必须使用中文"
            "\n- 请在报告末尾附上Markdown表格，整理报告中的关键要点"
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
            "capital_flow_report": report,
        }

    return capital_flow_analyst_node
