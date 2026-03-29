from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import build_instrument_context, get_news
from tradingagents.agents.utils.astock_tools import get_limit_updown


def create_market_sentiment_analyst(llm):
    def market_sentiment_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_limit_updown,
            get_news,
        ]

        system_message = (
            "你是 A 股市场情绪分析师。你的核心任务是判断当前市场的情绪状态和博弈强度。"
            "\n\n请依次调用以下工具获取数据："
            "\n1. get_limit_updown — 涨跌停统计（情绪温度计）"
            "\n2. get_news — 搜索与情绪/市场氛围相关的新闻"
            "\n\n分析框架："
            "\n1. 涨跌停数据（情绪温度计）"
            "\n   - 涨停家数 > 80: 市场亢奋，注意过热风险"
            "\n   - 涨停家数 < 20: 市场低迷，可能接近底部"
            "\n   - 跌停家数突增: 恐慌信号"
            "\n   - 涨停封板率: 高封板率 = 做多意愿强"
            "\n2. 市场宽度（从涨跌停数据推断）"
            "\n   - 涨停多跌停少 = 市场整体偏强"
            "\n   - 是否存在'指数涨个股跌'的虹吸效应"
            "\n3. 情绪周期定位"
            "\n   - 冰点 → 修复 → 升温 → 过热 → 分歧 → 退潮"
            "\n   - 根据涨跌停数据和新闻舆论判断当前处于哪个阶段"
            "\n\n输出要求："
            "\n- 明确给出情绪周期定位（冰点/修复/升温/过热/分歧/退潮）"
            "\n- 对目标个股：当前情绪环境是否有利于其上涨/下跌"
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
            "market_sentiment_report": report,
        }

    return market_sentiment_analyst_node
