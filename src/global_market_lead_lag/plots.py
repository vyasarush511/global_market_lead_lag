"""Plotting helpers for saved research outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_equity_curve(
    bt: pd.DataFrame,
    date_col: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a strategy-vs-benchmark equity curve."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = bt.copy()
    plot_data["strategy_equity"] = (1 + plot_data["strategy_return"]).cumprod()
    plot_data["buy_hold_equity"] = (1 + plot_data["buy_hold_return"]).cumprod()

    plt.figure(figsize=(12, 5))
    plt.plot(plot_data[date_col], plot_data["strategy_equity"], label="Strategy")
    plt.plot(plot_data[date_col], plot_data["buy_hold_equity"], label="Buy & Hold")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_drawdown_curve(
    bt: pd.DataFrame,
    date_col: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a strategy-vs-benchmark drawdown curve."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = bt.copy()
    strategy_equity = (1 + plot_data["strategy_return"]).cumprod()
    benchmark_equity = (1 + plot_data["buy_hold_return"]).cumprod()
    plot_data["strategy_drawdown"] = strategy_equity / strategy_equity.cummax() - 1
    plot_data["buy_hold_drawdown"] = benchmark_equity / benchmark_equity.cummax() - 1

    plt.figure(figsize=(12, 5))
    plt.plot(plot_data[date_col], plot_data["strategy_drawdown"], label="Strategy")
    plt.plot(plot_data[date_col], plot_data["buy_hold_drawdown"], label="Buy & Hold")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_yearly_bar_chart(
    yearly: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a grouped bar chart of yearly strategy and benchmark returns."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.bar(yearly["year"] - 0.2, yearly["strategy_return"], width=0.4, label="Strategy")
    plt.bar(yearly["year"] + 0.2, yearly["benchmark_return"], width=0.4, label="Buy & Hold")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Total Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
