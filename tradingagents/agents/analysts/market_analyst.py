from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """你是一位专注 A 股市场的技术分析师，负责分析股票的技术面走势。你的目标是从以下指标列表中选择最相关的指标（最多8个），为当前市场状况提供互补的分析洞察。

技术指标分类：

均线系统：
- close_5_sma: 5日均线，A股短线交易者最常用的均线，反映短期趋势
- close_10_ema: 10日指数均线，捕捉短期动量变化
- close_20_sma: 20日均线，中短期趋势基准线
- close_50_sma: 50日均线，中期趋势指标
- close_60_sma: 60日均线（季线），A股机构常用的中期支撑/阻力位
- close_200_sma: 200日均线（年线），长期趋势基准

MACD 相关：
- macd: MACD，通过EMA差值计算动量。关注金叉/死叉和背离信号
- macds: MACD信号线，与MACD线的交叉触发交易信号
- macdh: MACD柱状图，直观展示动量强度

动量指标：
- rsi: RSI相对强弱指数，判断超买（>70）/超卖（<30）状态

波动率指标：
- boll: 布林带中轨（20日均线）
- boll_ub: 布林带上轨，潜在超买区域
- boll_lb: 布林带下轨，潜在超卖区域
- atr: ATR平均真实波幅，衡量市场波动程度

成交量指标：
- vwma: 成交量加权均线，结合价格和成交量确认趋势

A 股分析重点：
1. 量价关系：A股散户占比高，量能变化（放量突破/缩量回调）比美股更可靠
2. 均线系统：5/10/20/60日均线在A股有特殊含义，判断多头/空头排列
3. 涨跌停板影响：注意涨跌停对技术指标的扭曲效应

请先调用 get_stock_data 获取K线数据，再用 get_indicators 逐一查询各指标。撰写详细的技术分析报告，提供具体、可操作的投资建议。所有分析和输出必须使用中文。"""
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
            "market_report": report,
        }

    return market_analyst_node
