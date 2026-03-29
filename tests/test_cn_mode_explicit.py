import pytest
import inspect


class TestCNModeExplicit:
    @pytest.mark.unit
    def test_create_trader_accepts_market_param(self):
        """create_trader should accept a market parameter."""
        from tradingagents.agents.trader.trader import create_trader
        sig = inspect.signature(create_trader)
        assert "market" in sig.parameters

    @pytest.mark.unit
    def test_trader_no_report_inference(self):
        """Trader should NOT use bool(capital_flow or market_sentiment) pattern."""
        from tradingagents.agents.trader.trader import create_trader
        source = inspect.getsource(create_trader)
        assert "bool(capital_flow or market_sentiment)" not in source

    @pytest.mark.unit
    def test_create_conservative_accepts_market_param(self):
        """Risk debators should accept market parameter."""
        from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
        sig = inspect.signature(create_conservative_debator)
        assert "market" in sig.parameters

    @pytest.mark.unit
    def test_create_aggressive_accepts_market_param(self):
        """Aggressive debator should accept market parameter."""
        from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
        sig = inspect.signature(create_aggressive_debator)
        assert "market" in sig.parameters

    @pytest.mark.unit
    def test_create_neutral_accepts_market_param(self):
        """Neutral debator should accept market parameter."""
        from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
        sig = inspect.signature(create_neutral_debator)
        assert "market" in sig.parameters

    @pytest.mark.unit
    def test_create_portfolio_manager_accepts_market_param(self):
        """Portfolio manager should accept market parameter."""
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        sig = inspect.signature(create_portfolio_manager)
        assert "market" in sig.parameters
