"""Tushare Pro data provider for A-share market data."""

import functools
import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import tushare as ts
from stockstats import wrap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache decorator
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".tradingagents-cn" / "cache"
_CACHE_TTL_HOURS = 24


def _cache_key(*args) -> str:
    raw = "|".join(str(a) for a in args)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cached(func):
    """Simple JSON file cache with 24h TTL."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(func.__name__, *args, *kwargs.values())
        cache_file = _CACHE_DIR / f"{key}.json"

        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=_CACHE_TTL_HOURS):
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass  # corrupted cache, re-fetch

        result = func(*args, **kwargs)

        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except (OSError, TypeError):
            pass  # non-fatal

        return result

    return wrapper


# ---------------------------------------------------------------------------
# Tushare API wrapper
# ---------------------------------------------------------------------------

_api_instance = None


def _get_api():
    global _api_instance
    if _api_instance is not None:
        return _api_instance

    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError(
            "未找到 Tushare token。请设置 TUSHARE_TOKEN 环境变量，"
            "或在 ~/.env.secrets 中配置。"
        )
    api_url = os.environ.get("TUSHARE_API_URL", "")
    if api_url:
        ts.set_token(token)
        pro = ts.pro_api(token)
        pro._DataApi__http_url = api_url  # Note: accesses private attr for custom API URL
        _api_instance = pro
    else:
        _api_instance = ts.pro_api(token)
    return _api_instance


def _exchange_for_code(code: str) -> str:
    """Determine exchange suffix from 6-digit A-share code."""
    if code.startswith("6"):
        return "SH"
    if code.startswith(("0", "3")):
        return "SZ"
    if code.startswith(("4", "8")):
        raise ValueError(
            f"暂不支持北交所股票（{code}），请输入沪深 A 股代码"
        )
    raise ValueError(f"无法识别的股票代码：{code}")


def _ts_code(symbol: str) -> str:
    """Convert plain 6-digit code to Tushare ts_code format (e.g. 600519.SH)."""
    code = symbol.strip().replace(".SH", "").replace(".SZ", "")
    exchange = _exchange_for_code(code)
    return f"{code}.{exchange}"


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------

def get_last_trading_date(date_str: str) -> str:
    """Find the most recent trading date on or before the given date.

    Uses Tushare trade_cal API to check if the date is a trading day.
    If not, returns the most recent trading day before it.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        The most recent trading date in YYYY-MM-DD format.
    """
    api = _get_api()
    end = date_str.replace("-", "")
    # Look back up to 10 days to find a trading day
    start_dt = pd.to_datetime(date_str) - timedelta(days=10)
    start = start_dt.strftime("%Y%m%d")

    df = api.trade_cal(start_date=start, end_date=end, is_open="1")
    if df is None or df.empty:
        return date_str  # fallback to original if API fails

    df = df.sort_values("cal_date", ascending=False)
    last_date = df.iloc[0]["cal_date"]
    # Convert YYYYMMDD to YYYY-MM-DD
    return f"{last_date[:4]}-{last_date[4:6]}-{last_date[6:]}"


# ---------------------------------------------------------------------------
# Core stock data
# ---------------------------------------------------------------------------

@cached
def get_stock_data_tushare(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Fetch OHLCV data from Tushare daily endpoint."""
    api = _get_api()
    ts_code = _ts_code(symbol)
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    df = api.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return f"未找到 {symbol} 在 {start_date} 到 {end_date} 期间的数据"

    df = df.sort_values("trade_date")
    df = df.rename(columns={
        "trade_date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "vol": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    csv_string = df[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(index=False)

    header = (
        f"# {ts_code} 股票数据 {start_date} 至 {end_date}\n"
        f"# 共 {len(df)} 条记录\n"
        f"# 数据获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# Technical indicators (reuses stockstats via Tushare OHLCV)
# ---------------------------------------------------------------------------

@cached
def get_indicators_tushare(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Compute a technical indicator using Tushare OHLCV + stockstats."""
    api = _get_api()
    ts_code = _ts_code(symbol)

    curr_date_dt = pd.to_datetime(curr_date)
    # Fetch enough history for long-term indicators
    start_dt = curr_date_dt - pd.DateOffset(years=2)
    start = start_dt.strftime("%Y%m%d")
    end = curr_date_dt.strftime("%Y%m%d")

    df = api.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return f"未找到 {symbol} 的历史数据"

    df = df.sort_values("trade_date")
    df = df.rename(columns={
        "trade_date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "vol": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])

    # Use stockstats for indicator calculation
    sdf = wrap(df)
    sdf["Date"] = sdf["Date"].dt.strftime("%Y-%m-%d")

    try:
        sdf[indicator]  # trigger stockstats calculation
    except (KeyError, AttributeError):
        logger.debug("不支持的指标 %s", indicator)
        return f"不支持的指标: {indicator}"

    curr_date_str = curr_date_dt.strftime("%Y-%m-%d")
    matching = sdf[sdf["Date"].str.startswith(curr_date_str)]

    if not matching.empty:
        value = matching[indicator].values[0]
        return f"{indicator} = {value} (日期: {curr_date_str})"
    return f"N/A: {curr_date_str} 非交易日"


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

@cached
def get_fundamentals_tushare(ticker: str, curr_date: str) -> str:
    """Comprehensive fundamental data for an A-share stock."""
    api = _get_api()
    ts_code = _ts_code(ticker)

    parts = []

    # Basic info
    basic = api.daily_basic(ts_code=ts_code, trade_date=curr_date.replace("-", ""))
    if basic is not None and not basic.empty:
        row = basic.iloc[0]
        parts.append(
            f"## 基本面快照 ({curr_date})\n"
            f"- 市盈率(PE): {row.get('pe', 'N/A')}\n"
            f"- 市净率(PB): {row.get('pb', 'N/A')}\n"
            f"- 总市值: {row.get('total_mv', 'N/A')} 万元\n"
            f"- 流通市值: {row.get('circ_mv', 'N/A')} 万元\n"
            f"- 换手率: {row.get('turnover_rate', 'N/A')}%\n"
        )

    # Income statement (latest quarter)
    income = api.income(ts_code=ts_code, period=curr_date[:4] + "0331")
    if income is None or income.empty:
        income = api.income(ts_code=ts_code, period=str(int(curr_date[:4]) - 1) + "1231")
    if income is not None and not income.empty:
        row = income.iloc[0]
        parts.append(
            f"## 利润表\n"
            f"- 营业收入: {row.get('revenue', 'N/A')}\n"
            f"- 净利润: {row.get('n_income', 'N/A')}\n"
            f"- 报告期: {row.get('end_date', 'N/A')}\n"
        )

    if not parts:
        return f"未找到 {ticker} 的基本面数据"
    return "\n".join(parts)


@cached
def get_balance_sheet_tushare(
    ticker: str,
    freq: str = "quarterly",
    curr_date: Optional[str] = None,
) -> str:
    """Balance sheet data from Tushare."""
    api = _get_api()
    ts_code = _ts_code(ticker)
    year = curr_date[:4] if curr_date else str(datetime.now().year)

    period = year + "1231" if freq == "annual" else year + "0331"
    df = api.balancesheet(ts_code=ts_code, period=period)
    if df is None or df.empty:
        # fallback to previous year
        df = api.balancesheet(ts_code=ts_code, period=str(int(year) - 1) + "1231")
    if df is None or df.empty:
        return f"未找到 {ticker} 的资产负债表数据"

    row = df.iloc[0]
    return (
        f"## 资产负债表 ({row.get('end_date', 'N/A')})\n"
        f"- 总资产: {row.get('total_assets', 'N/A')}\n"
        f"- 总负债: {row.get('total_liab', 'N/A')}\n"
        f"- 股东权益: {row.get('total_hldr_eqy_exc_min_int', 'N/A')}\n"
        f"- 货币资金: {row.get('money_cap', 'N/A')}\n"
    )


@cached
def get_cashflow_tushare(
    ticker: str,
    freq: str = "quarterly",
    curr_date: Optional[str] = None,
) -> str:
    """Cash flow statement from Tushare."""
    api = _get_api()
    ts_code = _ts_code(ticker)
    year = curr_date[:4] if curr_date else str(datetime.now().year)

    period = year + "1231" if freq == "annual" else year + "0331"
    df = api.cashflow(ts_code=ts_code, period=period)
    if df is None or df.empty:
        df = api.cashflow(ts_code=ts_code, period=str(int(year) - 1) + "1231")
    if df is None or df.empty:
        return f"未找到 {ticker} 的现金流量表数据"

    row = df.iloc[0]
    return (
        f"## 现金流量表 ({row.get('end_date', 'N/A')})\n"
        f"- 经营活动现金流: {row.get('n_cashflow_act', 'N/A')}\n"
        f"- 投资活动现金流: {row.get('n_cashflow_inv_act', 'N/A')}\n"
        f"- 筹资活动现金流: {row.get('n_cash_flows_fnc_act', 'N/A')}\n"
    )


@cached
def get_income_statement_tushare(
    ticker: str,
    freq: str = "quarterly",
    curr_date: Optional[str] = None,
) -> str:
    """Income statement from Tushare."""
    api = _get_api()
    ts_code = _ts_code(ticker)
    year = curr_date[:4] if curr_date else str(datetime.now().year)

    period = year + "1231" if freq == "annual" else year + "0331"
    df = api.income(ts_code=ts_code, period=period)
    if df is None or df.empty:
        df = api.income(ts_code=ts_code, period=str(int(year) - 1) + "1231")
    if df is None or df.empty:
        return f"未找到 {ticker} 的利润表数据"

    row = df.iloc[0]
    return (
        f"## 利润表 ({row.get('end_date', 'N/A')})\n"
        f"- 营业收入: {row.get('revenue', 'N/A')}\n"
        f"- 营业成本: {row.get('oper_cost', 'N/A')}\n"
        f"- 净利润: {row.get('n_income', 'N/A')}\n"
        f"- 归母净利润: {row.get('n_income_attr_p', 'N/A')}\n"
        f"- 基本每股收益: {row.get('basic_eps', 'N/A')}\n"
    )


# ---------------------------------------------------------------------------
# News (via Tavily)
# ---------------------------------------------------------------------------

def _get_tavily_client():
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError("请安装 tavily-python: pip install tavily-python")
    token = os.environ.get("TAVILY_API_KEY", "")
    if not token:
        raise RuntimeError("未找到 TAVILY_API_KEY 环境变量")
    return TavilyClient(api_key=token)


@cached
def get_news_tushare(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Search for stock-specific news via Tavily."""
    client = _get_tavily_client()
    ts_code = _ts_code(ticker)

    query = f"{ts_code} A股 最新消息 分析"
    results = client.search(query=query, max_results=8, search_depth="advanced")

    if not results.get("results"):
        return f"未找到 {ticker} 的相关新闻"

    parts = [f"## {ts_code} 相关新闻\n"]
    for i, r in enumerate(results["results"][:8], 1):
        parts.append(
            f"### {i}. {r.get('title', '无标题')}\n"
            f"来源: {r.get('url', 'N/A')}\n"
            f"{r.get('content', '无内容')[:500]}\n"
        )
    return "\n".join(parts)


@cached
def get_global_news_tushare(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 5,
) -> str:
    """Search for macro/policy news via Tavily."""
    client = _get_tavily_client()

    query = "中国 A股 宏观经济 政策 央行 证监会 最新动态"
    results = client.search(query=query, max_results=limit, search_depth="advanced")

    if not results.get("results"):
        return "未找到宏观新闻"

    parts = ["## 宏观经济与政策新闻\n"]
    for i, r in enumerate(results["results"][:limit], 1):
        parts.append(
            f"### {i}. {r.get('title', '无标题')}\n"
            f"来源: {r.get('url', 'N/A')}\n"
            f"{r.get('content', '无内容')[:500]}\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Insider transactions -> 大股东增减持
# ---------------------------------------------------------------------------

@cached
def get_insider_transactions_tushare(ticker: str) -> str:
    """Major shareholder changes (大股东增减持) from Tushare."""
    api = _get_api()
    ts_code = _ts_code(ticker)

    df = api.stk_holdertrade(ts_code=ts_code)
    if df is None or df.empty:
        return f"未找到 {ticker} 的大股东增减持数据"

    parts = [f"## {ts_code} 大股东增减持\n"]
    for _, row in df.head(10).iterrows():
        parts.append(
            f"- {row.get('holder_name', 'N/A')}: "
            f"{row.get('in_de', 'N/A')} "
            f"{row.get('change_vol', 'N/A')} 股 "
            f"(日期: {row.get('ann_date', 'N/A')})\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# A-share specific data
# ---------------------------------------------------------------------------

@cached
def get_northbound_flow_tushare(
    curr_date: str,
    look_back_days: int = 10,
) -> str:
    """Northbound capital flow (北向资金) from Tushare."""
    api = _get_api()
    end_dt = pd.to_datetime(curr_date)
    start_dt = end_dt - timedelta(days=look_back_days + 5)  # buffer for non-trading days
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    df = api.moneyflow_hsgt(start_date=start, end_date=end)
    if df is None or df.empty:
        return f"未找到 {curr_date} 附近的北向资金数据"

    df = df.sort_values("trade_date").tail(look_back_days)
    parts = ["## 北向资金流向\n"]
    for _, row in df.iterrows():
        parts.append(
            f"- {row.get('trade_date', 'N/A')}: "
            f"沪股通 {row.get('north_money', 'N/A')} 万元, "
            f"深股通 {row.get('south_money', 'N/A')} 万元\n"
        )
    return "\n".join(parts)


@cached
def get_limit_updown_tushare(curr_date: str) -> str:
    """Limit up/down stocks (涨跌停) from Tushare."""
    api = _get_api()
    trade_date = curr_date.replace("-", "")

    # Limit up
    df_up = api.limit_list_d(trade_date=trade_date, limit_type="U")
    # Limit down
    df_down = api.limit_list_d(trade_date=trade_date, limit_type="D")

    parts = [f"## 涨跌停统计 ({curr_date})\n"]

    if df_up is not None and not df_up.empty:
        parts.append(f"### 涨停 ({len(df_up)} 只)\n")
        for _, row in df_up.head(20).iterrows():
            parts.append(
                f"- {row.get('ts_code', 'N/A')} {row.get('name', '')}: "
                f"封板资金 {row.get('fd_amount', 'N/A')} 万\n"
            )
    else:
        parts.append("### 涨停: 无数据\n")

    if df_down is not None and not df_down.empty:
        parts.append(f"\n### 跌停 ({len(df_down)} 只)\n")
        for _, row in df_down.head(20).iterrows():
            parts.append(
                f"- {row.get('ts_code', 'N/A')} {row.get('name', '')}\n"
            )
    else:
        parts.append("\n### 跌停: 无数据\n")

    return "\n".join(parts)


@cached
def get_dragon_tiger_tushare(ticker: str, curr_date: str) -> str:
    """Dragon Tiger Board (龙虎榜) data from Tushare."""
    api = _get_api()
    ts_code = _ts_code(ticker)
    trade_date = curr_date.replace("-", "")

    df = api.top_list(trade_date=trade_date, ts_code=ts_code)
    if df is None or df.empty:
        # Try without ts_code filter to get market-wide data
        df = api.top_list(trade_date=trade_date)
        if df is not None and not df.empty:
            df = df[df["ts_code"] == ts_code]

    if df is None or df.empty:
        return f"未找到 {ticker} 在 {curr_date} 的龙虎榜数据（仅在异常波动时出现）"

    parts = [f"## {ts_code} 龙虎榜 ({curr_date})\n"]
    for _, row in df.iterrows():
        parts.append(
            f"- 原因: {row.get('reason', 'N/A')}\n"
            f"  买入额: {row.get('buy', 'N/A')} 万, "
            f"卖出额: {row.get('sell', 'N/A')} 万, "
            f"净买入: {row.get('net_buy', 'N/A')} 万\n"
        )
    return "\n".join(parts)


@cached
def get_block_trade_tushare(ticker: str, curr_date: str) -> str:
    """Block trade (大宗交易) data from Tushare."""
    api = _get_api()
    ts_code = _ts_code(ticker)
    end_dt = pd.to_datetime(curr_date)
    start_dt = end_dt - timedelta(days=30)
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    df = api.block_trade(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return f"未找到 {ticker} 近30天的大宗交易数据"

    df = df.sort_values("trade_date", ascending=False)
    parts = [f"## {ts_code} 大宗交易（近30天）\n"]
    for _, row in df.head(10).iterrows():
        parts.append(
            f"- {row.get('trade_date', 'N/A')}: "
            f"成交价 {row.get('price', 'N/A')} 元, "
            f"成交量 {row.get('vol', 'N/A')} 万股, "
            f"成交额 {row.get('amount', 'N/A')} 万元, "
            f"买方 {row.get('buyer', 'N/A')}, "
            f"卖方 {row.get('seller', 'N/A')}\n"
        )
    return "\n".join(parts)


@cached
def get_sector_performance_tushare(ticker: str, curr_date: str) -> str:
    """Sector/industry performance (板块联动) from Tushare."""
    api = _get_api()
    ts_code = _ts_code(ticker)

    # Get stock's industry classification
    member = api.stock_basic(ts_code=ts_code, fields="ts_code,name,industry")
    if member is None or member.empty:
        return f"未找到 {ticker} 的行业信息"

    industry = member.iloc[0].get("industry", "")
    stock_name = member.iloc[0].get("name", ticker)

    if not industry:
        return f"未找到 {ticker} ({stock_name}) 的行业分类"

    # Get all stocks in the same industry
    peers = api.stock_basic(industry=industry, fields="ts_code,name")
    if peers is None or peers.empty:
        return f"{industry} 行业无其他个股数据"

    # Get daily performance for peers on the trade date
    trade_date = curr_date.replace("-", "")
    peer_codes = ",".join(peers["ts_code"].tolist()[:20])  # limit to 20

    df = api.daily(ts_code=peer_codes, trade_date=trade_date)
    if df is None or df.empty:
        return f"{industry} 行业在 {curr_date} 无交易数据"

    df = df.merge(peers, on="ts_code", how="left")
    df = df.sort_values("pct_chg", ascending=False)

    parts = [
        f"## 板块联动分析\n"
        f"个股: {stock_name} ({ts_code})\n"
        f"所属行业: {industry}\n"
        f"同行业个股 {curr_date} 涨跌幅:\n"
    ]
    for _, row in df.iterrows():
        marker = "📍" if row["ts_code"] == ts_code else "  "
        parts.append(
            f"{marker} {row.get('name', 'N/A')} ({row['ts_code']}): "
            f"{row.get('pct_chg', 'N/A')}%\n"
        )
    return "\n".join(parts)


@cached
def get_margin_data_tushare(ticker: str, curr_date: str) -> str:
    """Margin trading data (融资融券) from Tushare."""
    api = _get_api()
    ts_code = _ts_code(ticker)
    end_dt = pd.to_datetime(curr_date)
    start_dt = end_dt - timedelta(days=15)
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    df = api.margin_detail(
        ts_code=ts_code,
        start_date=start,
        end_date=end,
    )
    if df is None or df.empty:
        return f"未找到 {ticker} 的融资融券数据"

    df = df.sort_values("trade_date")
    parts = [f"## {ts_code} 融资融券\n"]
    for _, row in df.tail(5).iterrows():
        parts.append(
            f"- {row.get('trade_date', 'N/A')}: "
            f"融资余额 {row.get('rzye', 'N/A')} 元, "
            f"融券余额 {row.get('rqye', 'N/A')} 元\n"
        )
    return "\n".join(parts)
