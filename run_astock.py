#!/usr/bin/env python3
"""A 股分析入口脚本。

用法:
    python run_astock.py --ticker 600519
    python run_astock.py --ticker 600519 --date 2026-03-28
"""

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------

def _load_env():
    """Load environment variables from multiple sources."""
    # Standard .env
    load_dotenv()
    # User secrets file
    secrets_path = Path.home() / ".env.secrets"
    if secrets_path.exists():
        load_dotenv(secrets_path)


def _validate_tokens():
    """Validate required API tokens exist before starting."""
    missing = []
    if not os.environ.get("TUSHARE_TOKEN"):
        missing.append("TUSHARE_TOKEN")
    if not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")

    # Check for LLM provider key (MiniMax or others)
    llm_keys = [
        "MINIMAX_API_KEY", "MINIMAX_CODING_PLAN_KEY",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    ]
    if not any(os.environ.get(k) for k in llm_keys):
        missing.append("LLM API Key (MINIMAX_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)")

    if missing:
        console.print(Panel(
            "\n".join(f"  [red]✗[/red] {k}" for k in missing),
            title="[red]缺少必要的环境变量[/red]",
            subtitle="请在 ~/.env.secrets 或环境变量中配置",
        ))
        sys.exit(1)

    console.print("[green]✓[/green] 所有 API token 验证通过")


# ---------------------------------------------------------------------------
# Ticker validation
# ---------------------------------------------------------------------------

def _validate_ticker(ticker: str) -> str:
    """Validate and normalize A-share ticker."""
    code = ticker.strip().replace(".SH", "").replace(".SZ", "")
    if not code.isdigit() or len(code) != 6:
        console.print(f"[red]错误: 无效的股票代码 '{ticker}'。请输入6位数字代码（如 600519）[/red]")
        sys.exit(1)
    if code.startswith(("4", "8")):
        console.print(f"[red]错误: 暂不支持北交所股票（{code}），请输入沪深 A 股代码[/red]")
        sys.exit(1)
    if code.startswith("6"):
        return code  # SH
    if code.startswith(("0", "3")):
        return code  # SZ
    console.print(f"[red]错误: 无法识别的股票代码 '{code}'[/red]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class ProgressCallback:
    """Track agent execution progress via rich console output."""

    STAGE_NAMES = {
        "Market Analyst": "技术分析师",
        "Social Analyst": "情绪分析师",
        "News Analyst": "新闻分析师",
        "Fundamentals Analyst": "基本面分析师",
        "Bull Researcher": "多头研究员",
        "Bear Researcher": "空头研究员",
        "Research Manager": "研究经理",
        "Trader": "交易员",
        "Aggressive Analyst": "激进型风控",
        "Conservative Analyst": "保守型风控",
        "Neutral Analyst": "中立型风控",
        "Portfolio Manager": "投资组合经理",
    }

    def __init__(self):
        self._current_stage = None

    def on_node_start(self, node_name: str):
        display_name = self.STAGE_NAMES.get(node_name, node_name)
        if "tools_" in node_name or "Msg Clear" in node_name:
            return  # skip tool/cleanup nodes
        self._current_stage = display_name
        console.print(f"  [cyan]▶[/cyan] {display_name} 分析中...")

    def on_node_end(self, node_name: str):
        display_name = self.STAGE_NAMES.get(node_name, node_name)
        if "tools_" in node_name or "Msg Clear" in node_name:
            return
        console.print(f"  [green]✓[/green] {display_name} 完成")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A 股多 Agent 分析系统")
    parser.add_argument("--ticker", required=True, help="A 股代码（如 600519）")
    parser.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--provider", default="minimax", help="LLM provider（默认 minimax）")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    # Load env and validate
    _load_env()
    _validate_tokens()

    ticker = _validate_ticker(args.ticker)
    raw_date = args.date or date.today().strftime("%Y-%m-%d")

    # Auto-fallback to last trading date
    from tradingagents.dataflows.tushare_provider import get_last_trading_date
    trade_date = get_last_trading_date(raw_date)
    if trade_date != raw_date:
        console.print(f"[yellow]📅 {raw_date} 非交易日，已自动调整为 {trade_date}[/yellow]")

    exchange = "SH" if ticker.startswith("6") else "SZ"
    display_code = f"{ticker}.{exchange}"

    console.print(Panel(
        f"股票代码: [bold]{display_code}[/bold]\n"
        f"分析日期: [bold]{trade_date}[/bold]\n"
        f"LLM: [bold]{args.provider}[/bold]",
        title="[bold blue]A 股多 Agent 分析系统[/bold blue]",
    ))

    # Build config for CN market
    from tradingagents.default_config import DEFAULT_CONFIG

    # Provider-specific backend URLs
    PROVIDER_URLS = {
        "minimax": "https://api.minimaxi.com/anthropic",
        "anthropic": "https://api.anthropic.com/",
        "openai": "https://api.openai.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1",
        "xai": "https://api.x.ai/v1",
        "ollama": "http://localhost:11434/v1",
    }

    config = {
        **DEFAULT_CONFIG,
        "market": "cn",
        "llm_provider": args.provider,
        "backend_url": PROVIDER_URLS.get(args.provider, DEFAULT_CONFIG["backend_url"]),
        "data_vendors": {
            "core_stock_apis": "tushare",
            "technical_indicators": "tushare",
            "fundamental_data": "tushare",
            "news_data": "tushare",
        },
    }

    if args.model:
        config["deep_think_llm"] = args.model
        config["quick_think_llm"] = args.model

    # Import and run
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    console.print("\n[bold]开始分析...[/bold]\n")
    start_time = datetime.now()

    graph = TradingAgentsGraph(
        debug=True,
        config=config,
    )

    final_state, signal = graph.propagate(ticker, trade_date)

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(f"\n[green]分析完成！[/green] 耗时 {elapsed:.1f} 秒\n")

    # Output report
    report = final_state.get("final_trade_decision", "无决策输出")
    console.print(Panel(report, title=f"[bold]{display_code} 分析报告[/bold]"))

    # Save to file
    report_dir = Path(f"reports/{ticker}")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"{trade_date}.md"
    report_file.write_text(
        f"# {display_code} 分析报告\n\n"
        f"日期: {trade_date}\n\n"
        f"---\n\n"
        f"{report}\n",
        encoding="utf-8",
    )
    console.print(f"[dim]报告已保存至 {report_file}[/dim]")


if __name__ == "__main__":
    main()
