"""A-share specific tool definitions for LangChain agents."""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_northbound_flow(
    curr_date: Annotated[str, "当前交易日期，格式 YYYY-mm-dd"],
    look_back_days: Annotated[int, "回看天数"] = 10,
) -> str:
    """
    获取北向资金（沪股通/深股通）流向数据。
    北向资金是外资通过港股通流入 A 股的资金，是判断外资态度的重要指标。
    Args:
        curr_date (str): 当前交易日期，YYYY-mm-dd
        look_back_days (int): 回看天数，默认10天
    Returns:
        str: 北向资金流向报告
    """
    return route_to_vendor("get_northbound_flow", curr_date, look_back_days)


@tool
def get_limit_updown(
    curr_date: Annotated[str, "当前交易日期，格式 YYYY-mm-dd"],
) -> str:
    """
    获取当日涨跌停统计数据。
    包括涨停股票列表、跌停股票列表、封板资金等。
    涨跌停是 A 股特有机制（每日涨跌幅限制10%/20%），反映市场情绪极端状态。
    Args:
        curr_date (str): 当前交易日期，YYYY-mm-dd
    Returns:
        str: 涨跌停统计报告
    """
    return route_to_vendor("get_limit_updown", curr_date)


@tool
def get_margin_data(
    ticker: Annotated[str, "股票代码"],
    curr_date: Annotated[str, "当前交易日期，格式 YYYY-mm-dd"],
) -> str:
    """
    获取个股融资融券数据。
    融资余额增加通常表示投资者看多，融券余额增加表示看空。
    Args:
        ticker (str): 股票代码，如 600519
        curr_date (str): 当前交易日期，YYYY-mm-dd
    Returns:
        str: 融资融券数据报告
    """
    return route_to_vendor("get_margin_data", ticker, curr_date)
