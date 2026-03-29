import pytest
from unittest.mock import MagicMock
import pandas as pd


class TestFindLatestTradingDay:
    @pytest.mark.unit
    def test_nontrading_day_shows_fallback_note(self):
        """On non-trading day, result should mention the actual trading day used."""
        from tradingagents.dataflows.tushare_provider import _find_latest_trading_day

        mock_api = MagicMock()
        mock_api.trade_cal.return_value = pd.DataFrame({
            "cal_date": ["20260325", "20260326", "20260327", "20260328", "20260329"],
            "is_open": [1, 1, 1, 0, 0],
        })

        result = _find_latest_trading_day(mock_api, "20260329")
        assert result == "20260327"

    @pytest.mark.unit
    def test_trading_day_returns_same(self):
        """On a trading day, should return the same date."""
        from tradingagents.dataflows.tushare_provider import _find_latest_trading_day

        mock_api = MagicMock()
        mock_api.trade_cal.return_value = pd.DataFrame({
            "cal_date": ["20260325", "20260326", "20260327"],
            "is_open": [1, 1, 1],
        })

        result = _find_latest_trading_day(mock_api, "20260327")
        assert result == "20260327"

    @pytest.mark.unit
    def test_api_failure_returns_original_date(self):
        """If trade_cal API fails, should fall back to original date."""
        from tradingagents.dataflows.tushare_provider import _find_latest_trading_day

        mock_api = MagicMock()
        mock_api.trade_cal.side_effect = Exception("API error")

        result = _find_latest_trading_day(mock_api, "20260329")
        assert result == "20260329"
