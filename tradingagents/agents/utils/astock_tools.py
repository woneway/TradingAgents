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
def get_dragon_tiger(
    ticker: Annotated[str, "股票代码"],
    curr_date: Annotated[str, "当前交易日期，格式 YYYY-mm-dd"],
) -> str:
    """
    获取龙虎榜数据。
    龙虎榜记录了当日异常波动个股的机构和游资买卖明细，是判断主力资金动向的重要参考。
    仅在个股出现异常波动（如涨跌停、振幅超过7%等）时才有数据。
    Args:
        ticker (str): 股票代码，如 600519
        curr_date (str): 当前交易日期，YYYY-mm-dd
    Returns:
        str: 龙虎榜数据报告
    """
    return route_to_vendor("get_dragon_tiger", ticker, curr_date)


@tool
def get_block_trade(
    ticker: Annotated[str, "股票代码"],
    curr_date: Annotated[str, "当前交易日期，格式 YYYY-mm-dd"],
) -> str:
    """
    获取大宗交易数据（近30天）。
    大宗交易是机构间的大额交易，折价或溢价成交反映机构对个股的态度。
    Args:
        ticker (str): 股票代码，如 600519
        curr_date (str): 当前交易日期，YYYY-mm-dd
    Returns:
        str: 大宗交易数据报告
    """
    return route_to_vendor("get_block_trade", ticker, curr_date)


@tool
def get_sector_performance(
    ticker: Annotated[str, "股票代码"],
    curr_date: Annotated[str, "当前交易日期，格式 YYYY-mm-dd"],
) -> str:
    """
    获取板块联动数据。
    分析个股所属行业板块中其他股票的涨跌情况，判断是否板块整体行情。
    如果同板块大部分个股上涨而目标个股未涨，可能存在补涨机会。
    Args:
        ticker (str): 股票代码，如 600519
        curr_date (str): 当前交易日期，YYYY-mm-dd
    Returns:
        str: 板块联动分析报告
    """
    return route_to_vendor("get_sector_performance", ticker, curr_date)


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
