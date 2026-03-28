from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_news, get_global_news


def create_policy_analyst(llm):
    def policy_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "你是一位 A 股政策分析师，专注于分析宏观政策、行业政策和监管动态对个股的影响。"
            "\n\n使用 get_news 搜索与个股相关的政策新闻，使用 get_global_news 获取宏观政策动态。"
            "\n\n分析重点："
            "\n1. **货币政策**：央行利率决议、MLF/LPR 调整、存准率变动、公开市场操作"
            "\n2. **财政政策**：减税降费、专项债发行、财政刺激计划"
            "\n3. **监管政策**：证监会新规（IPO 节奏、再融资、减持规定、退市制度）"
            "\n4. **行业政策**：产业扶持（新能源/芯片/AI）、行业整治（教育/地产/互联网）、碳中和相关"
            "\n5. **国际因素**：中美关系、出口管制、汇率政策、外资准入"
            "\n6. **重大会议**：两会、政治局会议、经济工作会议释放的信号"
            "\n\nA 股是典型的「政策市」，一个证监会周末发文可能导致周一全市场涨跌 3%。"
            "\n政策分析的核心是判断：当前政策环境对该个股是利好还是利空？力度如何？持续性如何？"
            "\n\n提供具体、可操作的政策面判断。所有分析和输出必须使用中文。"
            + """ 请在报告末尾附上Markdown表格，整理报告中的关键政策要点及其对个股的影响评估。"""
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
            "policy_report": report,
        }

    return policy_analyst_node
