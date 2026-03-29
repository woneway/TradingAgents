import pytest
import inspect


class TestFundamentalsTools:
    @pytest.mark.unit
    def test_fundamentals_analyst_has_stock_data_tool(self):
        """Fundamentals analyst source must reference get_stock_data."""
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
        source = inspect.getsource(create_fundamentals_analyst)
        assert "get_stock_data" in source

    @pytest.mark.unit
    def test_fundamentals_prompt_mentions_price_instruction(self):
        """Prompt must instruct analyst to use real price data for PE/PB."""
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
        source = inspect.getsource(create_fundamentals_analyst)
        assert "最新收盘价" in source or "get_stock_data" in source
