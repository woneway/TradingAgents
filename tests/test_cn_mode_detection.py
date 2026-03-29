"""Tests for CN mode analyst resolution logic."""

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph


CN_DEFAULT_ANALYSTS = [
    "market", "capital_flow", "sentiment", "news",
    "fundamentals", "policy", "sector_theme",
]


class FakeGraph:
    """Minimal stand-in to test _resolve_analysts without full init."""
    _resolve_analysts = TradingAgentsGraph._resolve_analysts


def _make(config):
    obj = FakeGraph()
    obj.config = config
    return obj


def test_cn_mode_forces_all_analysts():
    graph = _make({"market": "cn"})
    result = graph._resolve_analysts(["market", "news"])
    assert result == CN_DEFAULT_ANALYSTS


def test_cn_mode_ignores_user_selection():
    graph = _make({"market": "cn"})
    result = graph._resolve_analysts(["market", "social", "news", "fundamentals", "policy"])
    assert result == CN_DEFAULT_ANALYSTS


def test_us_mode_unchanged():
    graph = _make({"market": "us"})
    user_list = ["market", "social", "news", "fundamentals"]
    result = graph._resolve_analysts(user_list)
    assert result == ["market", "social", "news", "fundamentals"]
    assert result is not user_list


def test_cn_mode_with_empty_list():
    graph = _make({"market": "cn"})
    result = graph._resolve_analysts([])
    assert result == CN_DEFAULT_ANALYSTS
