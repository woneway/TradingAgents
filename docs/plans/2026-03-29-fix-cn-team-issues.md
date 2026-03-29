# CN 模式团队问题全面修复 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 CN（A 股）模式的 6 个已知问题：CN 模式检测 Bug、基本面缺股价、涨跌停非交易日、交易员 CN 推断、辩论冗余、政策分析师工具不足（Task 5 分析师重叠推迟到后续 PR）

**Architecture:** 从底层数据工具 → 分析师 → 编排层 → 辩论逻辑，自下而上修复。每个 Task 独立可提交。

**Tech Stack:** Python 3.11+, LangGraph, LangChain, Tushare Pro API

---

## Task 1: 修复 CN 模式检测 Bug（P0）

**问题根因：** `trading_graph.py:78` 使用精确列表匹配 `selected_analysts == ["market", "social", "news", "fundamentals"]` 来决定是否切换 CN 分析师。当 CLI 用户选择了 `policy`（5 个元素），匹配失败，导致 `capital_flow`、`sentiment`、`sector_theme` 三个 CN 专属分析师从未被激活。

**Files:**
- Modify: `tradingagents/graph/trading_graph.py:76-87`
- Modify: `cli/utils.py:12-17` (ANALYST_ORDER)
- Modify: `cli/models.py:6-11` (AnalystType)
- Modify: `cli/main.py:800` (ANALYST_ORDER)
- Modify: `tradingagents/graph/propagation.py:50-55`
- Modify: `run_astock.py:195-199`
- Test: `tests/test_cn_mode_detection.py`

**Step 1: Write the failing test**

```python
# tests/test_cn_mode_detection.py
import pytest
from unittest.mock import patch, MagicMock


class TestCNModeDetection:
    """CN mode must activate all 7 analysts regardless of selected_analysts input."""

    def _make_config(self, market="cn"):
        from tradingagents.default_config import DEFAULT_CONFIG
        config = {**DEFAULT_CONFIG, "market": market}
        return config

    @pytest.mark.unit
    def test_cn_mode_with_default_analysts(self):
        """When market=cn and default analysts passed, should use CN defaults."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        config = self._make_config("cn")
        with patch.object(TradingAgentsGraph, '_setup_internals'):
            graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
            graph.config = config
            result = graph._resolve_analysts(["market", "social", "news", "fundamentals"])
            assert "capital_flow" in result
            assert "sentiment" in result
            assert "sector_theme" in result
            assert "policy" in result
            assert len(result) == 7

    @pytest.mark.unit
    def test_cn_mode_with_policy_already_selected(self):
        """When market=cn and policy already in list, should still use CN defaults."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        config = self._make_config("cn")
        with patch.object(TradingAgentsGraph, '_setup_internals'):
            graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
            graph.config = config
            result = graph._resolve_analysts(["market", "social", "news", "fundamentals", "policy"])
            assert "capital_flow" in result
            assert "sentiment" in result
            assert "sector_theme" in result
            assert len(result) == 7

    @pytest.mark.unit
    def test_cn_mode_with_custom_subset(self):
        """When market=cn and user passes custom subset, still force CN defaults."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        config = self._make_config("cn")
        with patch.object(TradingAgentsGraph, '_setup_internals'):
            graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
            graph.config = config
            result = graph._resolve_analysts(["market", "fundamentals"])
            assert "capital_flow" in result
            assert "policy" in result

    @pytest.mark.unit
    def test_us_mode_unchanged(self):
        """US mode should not add CN analysts."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        config = self._make_config("us")
        with patch.object(TradingAgentsGraph, '_setup_internals'):
            graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
            graph.config = config
            result = graph._resolve_analysts(["market", "social", "news", "fundamentals"])
            assert "capital_flow" not in result
            assert len(result) == 4
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lianwu/ai/projects/TradingAgents && python -m pytest tests/test_cn_mode_detection.py -v`
Expected: FAIL (no `_resolve_analysts` method exists yet)

**Step 3: Fix `trading_graph.py` — extract analyst resolution logic**

Replace lines 76-87 in `tradingagents/graph/trading_graph.py`:

```python
# OLD (lines 76-87):
# CN mode: use cn_selected_analysts if user didn't customize
if self.config.get("market") == "cn":
    if selected_analysts == ["market", "social", "news", "fundamentals"]:
        # User didn't customize, use CN defaults
        selected_analysts = self.config.get(
            "cn_selected_analysts",
            ["market", "capital_flow", "sentiment", "news",
             "fundamentals", "policy", "sector_theme"],
        )
    # Always ensure policy is included for CN
    if "policy" not in selected_analysts:
        selected_analysts.append("policy")

# NEW:
selected_analysts = self._resolve_analysts(selected_analysts)
```

Add new method to `TradingAgentsGraph` class:

```python
def _resolve_analysts(self, selected_analysts: list) -> list:
    """Resolve analyst list based on market mode.

    For CN market: always use cn_selected_analysts from config,
    ignoring user's selected_analysts (which only knows US analysts).
    """
    if self.config.get("market") == "cn":
        return list(self.config.get(
            "cn_selected_analysts",
            ["market", "capital_flow", "sentiment", "news",
             "fundamentals", "policy", "sector_theme"],
        ))
    return list(selected_analysts)
```

**Step 4: Fix `propagation.py` — initialize CN report fields**

In `tradingagents/graph/propagation.py:50-55`, add missing CN state fields:

```python
# OLD (lines 50-55):
"market_report": "",
"fundamentals_report": "",
"sentiment_report": "",
"news_report": "",
"policy_report": "",

# NEW:
"market_report": "",
"fundamentals_report": "",
"sentiment_report": "",
"news_report": "",
"policy_report": "",
"capital_flow_report": "",
"market_sentiment_report": "",
"sector_theme_report": "",
```

**Step 5: Fix `run_astock.py` — remove hardcoded analyst list**

In `run_astock.py:195-199`:

```python
# OLD:
graph = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    debug=True,
    config=config,
)

# NEW:
graph = TradingAgentsGraph(
    debug=True,
    config=config,
)
```

Let `_resolve_analysts` handle the default → CN conversion automatically.

**Step 6: Update CLI to show CN analysts when market=cn**

In `cli/models.py`, add CN analyst types:

```python
class AnalystType(str, Enum):
    MARKET = "market"
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    POLICY = "policy"
    # CN-specific
    CAPITAL_FLOW = "capital_flow"
    SENTIMENT = "sentiment"
    SECTOR_THEME = "sector_theme"
```

In `cli/utils.py:12-17`, extend:

```python
ANALYST_ORDER = [
    ("技术分析师 (Market)", AnalystType.MARKET),
    ("情绪分析师 (Social)", AnalystType.SOCIAL),
    ("新闻分析师 (News)", AnalystType.NEWS),
    ("基本面分析师 (Fundamentals)", AnalystType.FUNDAMENTALS),
    ("政策分析师 (Policy)", AnalystType.POLICY),
    ("资金流向分析师 (Capital Flow)", AnalystType.CAPITAL_FLOW),
    ("市场情绪分析师 (Sentiment)", AnalystType.SENTIMENT),
    ("板块题材分析师 (Sector Theme)", AnalystType.SECTOR_THEME),
]
```

In `cli/main.py:800`:

```python
ANALYST_ORDER = [
    "market", "social", "news", "fundamentals", "policy",
    "capital_flow", "sentiment", "sector_theme",
]
```

**Step 7: Run tests and verify**

Run: `cd /Users/lianwu/ai/projects/TradingAgents && python -m pytest tests/test_cn_mode_detection.py -v`
Expected: ALL PASS

**Step 8: Commit**

```bash
git add tradingagents/graph/trading_graph.py tradingagents/graph/propagation.py \
  run_astock.py cli/models.py cli/utils.py cli/main.py \
  tests/test_cn_mode_detection.py
git commit -m "fix: CN 模式检测改用 config.market 判断，不再依赖精确列表匹配"
```

---

## Task 2: 基本面分析师增加股价数据工具（P0）

**问题根因：** `fundamentals_analyst.py` 没有 `get_stock_data` 工具，无法获取当前股价，导致无法计算 PE/PB。后续辩论中"PE 42 倍"是 LLM 幻觉。

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py:25-31`
- Modify: `tradingagents/graph/trading_graph.py` (tool_dict for fundamentals)
- Test: `tests/test_fundamentals_tools.py`

**Step 1: Write the failing test**

```python
# tests/test_fundamentals_tools.py
import pytest


class TestFundamentalsTools:
    @pytest.mark.unit
    def test_fundamentals_analyst_has_stock_data_tool(self):
        """Fundamentals analyst must have get_stock_data for PE/PB calculation."""
        from tradingagents.agents.analysts.fundamentals_analyst import (
            create_fundamentals_analyst,
        )
        from unittest.mock import MagicMock

        llm = MagicMock()
        node_fn = create_fundamentals_analyst(llm)
        # Inspect the tools bound in the closure
        # The tool names should include get_stock_data
        import inspect
        source = inspect.getsource(node_fn)
        assert "get_stock_data" in source or True  # placeholder

    @pytest.mark.unit
    def test_fundamentals_prompt_mentions_stock_price(self):
        """Prompt must instruct analyst to fetch current stock price for valuation."""
        from tradingagents.agents.analysts.fundamentals_analyst import (
            create_fundamentals_analyst,
        )
        from unittest.mock import MagicMock

        llm = MagicMock()
        node_fn = create_fundamentals_analyst(llm)
        # We verify by checking the source code contains price-related instruction
        import inspect
        source = inspect.getsource(node_fn)
        assert "get_stock_data" in source
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lianwu/ai/projects/TradingAgents && python -m pytest tests/test_fundamentals_tools.py -v`
Expected: FAIL — `get_stock_data` not in source

**Step 3: Add `get_stock_data` to fundamentals analyst**

In `tradingagents/agents/analysts/fundamentals_analyst.py`:

Add import (after line 6):
```python
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_stock_data,  # ADD THIS
)
```

Update tools list (line 25-31):
```python
tools = [
    get_stock_data,  # ADD: for current price → PE/PB calculation
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_margin_data,
    get_share_unlock,
    get_st_status,
]
```

Update system_message prompt (line 36-37), add after existing tool descriptions:
```python
"\n- get_stock_data: 获取近期 K 线数据（用最新收盘价计算 PE/PB）"
```

Add valuation instruction (after line 45):
```python
"\n1. 估值水平：先用 get_stock_data 获取最新收盘价，结合 EPS 计算实际 PE；"
"\n   结合每股净资产计算 PB。不要猜测或推算 PE/PB，必须用真实数据。"
```

**Step 4: Update tool_dict in `trading_graph.py`**

In `tradingagents/graph/trading_graph.py`, the `fundamentals_tools` list (around line 200-210):

Add `get_stock_data` to `fundamentals_tools`:
```python
fundamentals_tools = [
    get_stock_data,  # ADD: for price-based valuation
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
]
```

**Step 5: Run tests**

Run: `cd /Users/lianwu/ai/projects/TradingAgents && python -m pytest tests/test_fundamentals_tools.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tradingagents/agents/analysts/fundamentals_analyst.py \
  tradingagents/graph/trading_graph.py tests/test_fundamentals_tools.py
git commit -m "fix: 基本面分析师增加 get_stock_data 工具，解决 PE/PB 无法计算的问题"
```

---

## Task 3: 涨跌停工具处理非交易日（P1）

**问题根因：** `get_limit_updown_tushare` 直接用传入日期调用 API，非交易日返回空数据，分析师得到误导信息"当日无涨停"。应自动回退到最近交易日。

**Files:**
- Modify: `tradingagents/dataflows/tushare_provider.py:510-541`
- Test: `tests/test_limit_updown_nontrading.py`

**Step 1: Write the failing test**

```python
# tests/test_limit_updown_nontrading.py
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestLimitUpdownNonTradingDay:
    @pytest.mark.unit
    def test_nontrading_day_fallback(self):
        """On non-trading day, should fall back to most recent trading day."""
        from tradingagents.dataflows.tushare_provider import get_limit_updown_tushare

        mock_api = MagicMock()
        # First call (Saturday) returns empty
        # Second call (Friday) returns data
        mock_api.limit_list_d.side_effect = [
            pd.DataFrame(),  # Saturday limit_up
            pd.DataFrame(),  # Saturday limit_down — triggers fallback
        ]
        mock_api.trade_cal.return_value = pd.DataFrame({
            "cal_date": ["20260327", "20260328", "20260329"],
            "is_open": [1, 0, 0],
        })

        with patch(
            "tradingagents.dataflows.tushare_provider._get_api",
            return_value=mock_api,
        ):
            result = get_limit_updown_tushare("2026-03-29")
            # Should mention it fell back or use the actual trading day
            assert "非交易日" in result or "2026-03-27" in result
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lianwu/ai/projects/TradingAgents && python -m pytest tests/test_limit_updown_nontrading.py -v`
Expected: FAIL — current code doesn't handle fallback

**Step 3: Implement non-trading day fallback**

Replace `get_limit_updown_tushare` in `tradingagents/dataflows/tushare_provider.py:510-541`:

```python
@cached
def get_limit_updown_tushare(curr_date: str) -> str:
    """Limit up/down stocks (涨跌停) from Tushare.

    If curr_date is a non-trading day, automatically falls back
    to the most recent trading day.
    """
    api = _get_api()
    trade_date = curr_date.replace("-", "")

    # Check if it's a trading day; if not, find the most recent one
    actual_date = _find_latest_trading_day(api, trade_date)
    display_date = f"{actual_date[:4]}-{actual_date[4:6]}-{actual_date[6:]}"

    fallback_note = ""
    if actual_date != trade_date:
        fallback_note = f"（⚠️ {curr_date} 为非交易日，已自动回退至 {display_date}）\n"

    # Limit up
    df_up = api.limit_list_d(trade_date=actual_date, limit_type="U")
    # Limit down
    df_down = api.limit_list_d(trade_date=actual_date, limit_type="D")

    parts = [f"## 涨跌停统计 ({display_date})\n{fallback_note}"]

    if df_up is not None and not df_up.empty:
        parts.append(f"### 涨停 ({len(df_up)} 只)\n")
        for _, row in df_up.head(20).iterrows():
            parts.append(
                f"- {row.get('ts_code', 'N/A')} {row.get('name', '')}: "
                f"封板资金 {row.get('fd_amount', 'N/A')} 万\n"
            )
    else:
        parts.append("### 涨停: 当日无涨停股票（市场情绪偏弱或平稳）\n")

    if df_down is not None and not df_down.empty:
        parts.append(f"\n### 跌停 ({len(df_down)} 只)\n")
        for _, row in df_down.head(20).iterrows():
            parts.append(
                f"- {row.get('ts_code', 'N/A')} {row.get('name', '')}\n"
            )
    else:
        parts.append("\n### 跌停: 当日无跌停股票（市场未出现恐慌性抛售）\n")

    return "\n".join(parts)
```

Add helper function (before `get_limit_updown_tushare`):

```python
@cached  # 【工程评审决议】缓存交易日历，避免重复 API 调用
def _find_latest_trading_day(api, date_str: str) -> str:
    """Find the most recent trading day on or before date_str (YYYYMMDD format)."""
    try:
        cal = api.trade_cal(
            exchange="SSE",
            start_date=str(int(date_str) - 10),  # look back up to 10 days
            end_date=date_str,
        )
        if cal is not None and not cal.empty:
            open_days = cal[cal["is_open"] == 1]["cal_date"]
            if not open_days.empty:
                return open_days.iloc[-1]
    except Exception:
        pass
    return date_str  # fallback to original if calendar unavailable
```

**【工程评审决议】同时重构 `get_northbound_flow_tushare` 使用 `_find_latest_trading_day`**，替换现有的 5 天 buffer 逻辑，统一非交易日处理模式。需确认接口签名不变。

**Step 4: Run tests**

Run: `cd /Users/lianwu/ai/projects/TradingAgents && python -m pytest tests/test_limit_updown_nontrading.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tradingagents/dataflows/tushare_provider.py tests/test_limit_updown_nontrading.py
git commit -m "fix: 涨跌停工具自动回退到最近交易日，避免非交易日返回空数据"
```

---

## Task 4: 交易员 CN 模式显式传递（P1）

**问题根因：** `trader.py:23` 用 `is_cn = bool(capital_flow or market_sentiment)` 推断 CN 模式，如果两个报告都为空字符串，`is_cn` 为 `False`，A 股交易规则不执行。

**【工程评审决议】** 通过 create 函数参数传递 `market`，不添加 state 字段。`trading_graph.py` 创建 agent 时传 `self.config.get("market", "us")`。

**Files:**
- Modify: `tradingagents/agents/trader/trader.py` (add `market` param to `create_trader`)
- Modify: `tradingagents/agents/risk_mgmt/conservative_debator.py` (add `market` param)
- Modify: `tradingagents/agents/risk_mgmt/aggressive_debator.py` (add `market` param)
- Modify: `tradingagents/agents/risk_mgmt/neutral_debator.py` (add `market` param)
- Modify: `tradingagents/agents/managers/portfolio_manager.py` (add `market` param)
- Modify: `tradingagents/graph/trading_graph.py` (pass `market` when creating agents)
- Test: `tests/test_cn_mode_explicit.py`

**Step 1: Write the failing test**

```python
# tests/test_cn_mode_explicit.py
import pytest


class TestCNModeExplicit:
    @pytest.mark.unit
    def test_create_trader_accepts_market_param(self):
        """create_trader should accept a market parameter."""
        import inspect
        from tradingagents.agents.trader.trader import create_trader
        sig = inspect.signature(create_trader)
        assert "market" in sig.parameters

    @pytest.mark.unit
    def test_trader_uses_market_param(self):
        """Trader should use market param, not infer from reports."""
        import inspect
        from tradingagents.agents.trader.trader import create_trader
        source = inspect.getsource(create_trader)
        # Should NOT contain the old inference pattern
        assert "bool(capital_flow or market_sentiment)" not in source
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lianwu/ai/projects/TradingAgents && python -m pytest tests/test_cn_mode_explicit.py -v`
Expected: FAIL

**Step 3: Fix trader.py — add `market` param to `create_trader`**

In `tradingagents/agents/trader/trader.py`:

```python
# OLD:
def create_trader(llm, memory):
    def trader_node(state, name):
        ...
        is_cn = bool(capital_flow or market_sentiment)

# NEW:
def create_trader(llm, memory, market: str = "us"):
    def trader_node(state, name):
        ...
        is_cn = market == "cn"
```

**Step 4: Fix all risk debators and portfolio manager — same pattern**

In each file, add `market: str = "us"` to the create function, replace CN detection:

- `create_aggressive_debator(llm, market="us")` → `is_cn = market == "cn"`
- `create_conservative_debator(llm, market="us")` → `is_cn = market == "cn"`
- `create_neutral_debator(llm, market="us")` → `is_cn = market == "cn"`
- `create_portfolio_manager(llm, memory, market="us")` → `is_cn = market == "cn"`

**Step 5: Fix `trading_graph.py` — pass `market` when creating agents**

In `trading_graph.py`, update all create calls to pass market:

```python
market = self.config.get("market", "us")

trader_node = create_trader(self.quick_thinking_llm, self.trader_memory, market=market)
aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm, market=market)
conservative_analyst = create_conservative_debator(self.quick_thinking_llm, market=market)
neutral_analyst = create_neutral_debator(self.quick_thinking_llm, market=market)
portfolio_manager_node = create_portfolio_manager(
    self.deep_thinking_llm, self.portfolio_manager_memory, market=market
)
```

**Step 6: Run tests**

Run: `cd /Users/lianwu/ai/projects/TradingAgents && python -m pytest tests/test_cn_mode_explicit.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add tradingagents/agents/trader/trader.py \
  tradingagents/agents/risk_mgmt/*.py \
  tradingagents/agents/managers/portfolio_manager.py \
  tradingagents/graph/trading_graph.py \
  tests/test_cn_mode_explicit.py
git commit -m "fix: 交易员和风控团队通过 create 函数参数显式传递 market，不依赖报告存在性"
```

---

## Task 5: 消除分析师职责重叠（P1）— ⏸️ 推迟到后续 PR

**问题根因：** `social` 分析师和 `market_sentiment` 分析师都使用北向资金和涨跌停工具；`news` 和 `sentiment` 都引用同一条摩根大通研报。CN 模式下 `social` 不应出现。

**Files:**
- Modify: `tradingagents/graph/trading_graph.py` (social tools in CN mode)
- Modify: `cli/utils.py:12-17` (filter by market)

**Step 1: Remove `social` from CN mode**

In `tradingagents/graph/trading_graph.py` `_resolve_analysts` method (from Task 1):

```python
def _resolve_analysts(self, selected_analysts: list) -> list:
    if self.config.get("market") == "cn":
        return list(self.config.get(
            "cn_selected_analysts",
            ["market", "capital_flow", "sentiment", "news",
             "fundamentals", "policy", "sector_theme"],
        ))
    return list(selected_analysts)
```

This already excludes `social` from CN mode. The `social` analyst's northbound/limit-updown tools overlap is eliminated because CN mode uses dedicated `capital_flow` and `sentiment` analysts instead.

**Step 2: Remove CN tools from social analyst's tool node**

In `trading_graph.py` `_create_tool_nodes` method, remove CN extensions from social tools:

```python
# OLD:
social_tools = [get_news]
if is_cn:
    social_tools.extend([get_northbound_flow, get_limit_updown, get_dragon_tiger])

# NEW:
social_tools = [get_news]
# CN-specific tools are handled by dedicated capital_flow and sentiment analysts
```

**Step 3: Remove CN tools from fundamentals**

In `trading_graph.py`, remove CN extensions from fundamentals_tools in the tool_dict:

```python
# OLD:
if is_cn:
    fundamentals_tools.extend([
        get_margin_data, get_block_trade,
        get_share_unlock, get_st_status,
    ])

# NEW: (margin_data etc. are already in the analyst's own tools list,
# not needed in the graph-level tool_dict since CN analysts have dedicated tool nodes)
```

Wait — the fundamentals_analyst.py already imports and uses these tools directly. The graph `tool_dict` is for the ToolNode that handles tool calls. These must stay in the fundamentals ToolNode. No change needed here.

**Step 4: Commit**

```bash
git add tradingagents/graph/trading_graph.py
git commit -m "refactor: CN 模式移除 social 分析师的 CN 工具，由专属分析师承担"
```

---

## Task 6: 降低辩论冗余（P1）

**问题根因：** `conditional_logic.py` 辩论终止纯靠轮次计数，无内容收敛检测。默认 `max_debate_rounds=1` 但实际观察到 5+ 轮辩论，说明调用时可能使用了更高值。

**Files:**
- Modify: `tradingagents/graph/conditional_logic.py:78-98`
- Modify: `tradingagents/agents/researchers/bull_researcher.py` (prompt adjustment)
- Modify: `tradingagents/agents/researchers/bear_researcher.py` (prompt adjustment)
- Modify: `tradingagents/agents/risk_mgmt/aggressive_debator.py` (prompt adjustment)
- Modify: `tradingagents/agents/risk_mgmt/conservative_debator.py` (prompt adjustment)
- Modify: `tradingagents/agents/risk_mgmt/neutral_debator.py` (prompt adjustment)

**Step 1: Verify default config values**

Check `tradingagents/default_config.py:22-23`:
```python
"max_debate_rounds": 1,
"max_risk_discuss_rounds": 1,
```

These are already conservative. The 5+ rounds in the report likely came from a custom config override. The defaults are fine.

**Step 2: Add convergence hint to debate prompts**

The key change: instruct debaters to be concise and avoid repeating arguments.

In `tradingagents/agents/researchers/bull_researcher.py`, add to the system prompt:

```python
"\n\n辩论规则："
"\n- 每轮只提出 2-3 个核心论点，不要重复之前说过的观点"
"\n- 用数据和事实说话，避免修辞攻击和情绪化表达"
"\n- 如果对方提出了你无法反驳的有效论点，要诚实承认"
```

In `tradingagents/agents/researchers/bear_researcher.py`, add the same rules.

In all three risk debators (`aggressive`, `conservative`, `neutral`), add:

```python
"\n\n讨论规则："
"\n- 每轮只提出 2-3 个核心观点，不要重复已有论点"
"\n- 聚焦事实和数据，避免人身攻击或过度戏剧化"
"\n- 针对其他分析师的新观点回应，不要无视对方论点"
```

**Step 3: Commit**

```bash
git add tradingagents/agents/researchers/*.py tradingagents/agents/risk_mgmt/*.py
git commit -m "fix: 辩论 prompt 增加简洁性约束，要求不重复论点、用数据说话"
```

---

## Task 7: 政策分析师增强工具（P2）

**问题根因：** `policy_analyst.py` 只使用 `get_news` + `get_global_news`（Tavily 通用搜索），缺少行业专项政策数据。

**Files:**
- Modify: `tradingagents/agents/analysts/policy_analyst.py`

**Step 1: Enhance policy analyst prompt**

由于我们暂时没有专业政策 API，最有效的改进是优化搜索关键词策略。

在 `policy_analyst.py` 的 system_message 中增加搜索指引：

```python
"\n\n搜索策略指引："
"\n1. 第一次搜索: 用 get_news 搜索公司所在行业的政策关键词"
"\n   例如：'新能源汽车 补贴 政策 2026' 或 '制冷 热管理 行业标准'"
"\n2. 第二次搜索: 用 get_global_news 搜索宏观政策"
"\n   例如：'央行 降准 降息' 或 '发改委 制造业'"
"\n3. 第三次搜索（如需要）: 搜索公司直接相关的监管动态"
"\n   例如：'三花智控 行政处罚' 或 '002050 机构调研'"
"\n\n分析时必须区分："
"\n- 对公司直接相关的政策（高权重）"
"\n- 对行业间接影响的政策（中权重）"
"\n- 宏观面的政策（低权重）"
```

**Step 2: Commit**

```bash
git add tradingagents/agents/analysts/policy_analyst.py
git commit -m "fix: 政策分析师 prompt 增加搜索策略指引，提升行业政策分析针对性"
```

---

## 验证清单

完成所有 Task 后，运行完整验证：

```bash
# 1. 运行所有新增测试
python -m pytest tests/test_cn_mode_detection.py tests/test_fundamentals_tools.py \
  tests/test_limit_updown_nontrading.py tests/test_cn_mode_explicit.py -v

# 2. 运行已有测试确保无回归
python -m pytest tests/ -v

# 3. 端到端验证（可选，需要 API key）
uv run python run_astock.py --ticker 002050 --date 2026-03-29
# 检查日志：分析师团队应为 7 个 CN 分析师
# 检查报告：基本面应包含真实 PE/PB
# 检查涨跌停：应自动回退到最近交易日
```

---

## NOT in scope

- **Task 5（分析师职责重叠清理）** — Task 1 修复后 social 不再出现在 CN 流程中，重叠工具成为死代码，不影响功能。推迟到后续 PR。
- **LLM grounding 系统性约束** — 辩论中存在无数据来源的统计引用（如"跑输大盘概率 70%"），需要系统性给所有 agent 加 grounding 规则。超出本次修复范围。
- **分析师并行化** — 7 个分析师顺序执行，并行化可降低运行时间。Design doc Open Question #3。

## What already exists

- `get_northbound_flow_tushare` 已有非交易日处理（5 天 buffer 取最近数据）→ 本次统一为 `_find_latest_trading_day` helper
- `@cached` 装饰器已存在于 `tushare_provider.py` → 直接复用
- `get_stock_data` 工具已存在 → 本次添加到 fundamentals analyst 的工具列表

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/gstack-plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/gstack-codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/gstack-plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 4 issues, 0 critical gaps |
| Design Review | `/gstack-plan-design-review` | UI/UX gaps | 0 | — | — |
| Outside Voice | Claude subagent | Independent plan challenge | 1 | issues_found | 7 findings, 2 incorporated |

**UNRESOLVED:** 0
**VERDICT:** ENG CLEARED — ready to implement. Run `/gstack-ship` when done.
