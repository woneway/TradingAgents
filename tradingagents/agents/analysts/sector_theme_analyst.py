from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import build_instrument_context, get_news
from tradingagents.agents.utils.astock_tools import (
    get_sector_performance,
    get_concept_stocks,
)


def create_sector_theme_analyst(llm):
    def sector_theme_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_sector_performance,
            get_concept_stocks,
            get_news,
        ]

        system_message = (
            "你是 A 股板块题材分析师。你的核心任务是分析个股的题材属性和板块轮动位置。"
            "\n\n请依次调用以下工具获取数据："
            "\n1. get_concept_stocks — 查询个股所属概念板块"
            "\n2. get_sector_performance — 查询同行业个股涨跌表现"
            "\n3. get_news — 搜索与板块/题材相关的最新消息和催化剂"
            "\n\n分析框架："
            "\n1. 题材归属"
            "\n   - 目标个股属于哪些概念板块？"
            "\n   - 这些概念的当前市场热度如何？"
            "\n   - 是否有新的催化剂（政策、事件、技术突破）？"
            "\n2. 板块强度"
            "\n   - 同板块个股整体涨跌情况"
            "\n   - 板块龙头是谁？目标股在板块中的位置？"
            "\n   - 板块是否处于启动期/加速期/高潮期/退潮期？"
            "\n3. 轮动判断"
            "\n   - 当前市场主线是什么？（大金融/科技/消费/周期？）"
            "\n   - 目标个股所在板块与主线的关系？"
            "\n   - 是否存在补涨机会或已经滞涨？"
            "\n\n输出要求："
            "\n- 判断目标个股在板块中的地位（龙头/跟风/边缘）"
            "\n- 判断所属题材的生命周期阶段（启动/加速/高潮/退潮）"
            "\n- 给出题材维度的交易建议"
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
            "sector_theme_report": report,
        }

    return sector_theme_analyst_node
