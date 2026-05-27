"""Performance and diagnostic metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_stats(
    returns_series: pd.Series,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate common return, risk, and drawdown metrics."""

    returns = returns_series.dropna()
    if returns.empty:
        return pd.Series(
            {
                "total_return": np.nan,
                "annual_return": np.nan,
                "annual_volatility": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "win_rate": np.nan,
            }
        )

    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    annual_return = (
        equity.iloc[-1] ** (periods_per_year / len(returns)) - 1
        if equity.iloc[-1] > 0
        else np.nan
    )
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_volatility if annual_volatility != 0 else np.nan
    max_drawdown = (equity / equity.cummax() - 1).min()

    return pd.Series(
        {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": (returns > 0).mean(),
        }
    )


def strategy_diagnostics(bt: pd.DataFrame) -> pd.Series:
    """Summarize exposure, turnover, and invested-day behavior."""

    invested = bt["signal"] != 0
    traded = bt["turnover"] > 0
    invested_returns = bt.loc[invested, "strategy_return"]

    return pd.Series(
        {
            "days": len(bt),
            "invested_days": invested.sum(),
            "exposure": invested.mean(),
            "long_days": (bt["signal"] == 1).sum(),
            "short_days": (bt["signal"] == -1).sum(),
            "cash_days": (bt["signal"] == 0).sum(),
            "num_trades": traded.sum(),
            "avg_daily_turnover": bt["turnover"].mean(),
            "invested_win_rate": (invested_returns > 0).mean()
            if not invested_returns.empty
            else np.nan,
            "avg_invested_return": invested_returns.mean()
            if not invested_returns.empty
            else np.nan,
            "median_invested_return": invested_returns.median()
            if not invested_returns.empty
            else np.nan,
            "best_day": bt["strategy_return"].max(),
            "worst_day": bt["strategy_return"].min(),
        }
    )


def yearly_performance(
    bt: pd.DataFrame,
    date_col: str,
    strategy_col: str = "strategy_return",
    benchmark_col: str = "buy_hold_return",
) -> pd.DataFrame:
    """Calculate strategy and benchmark statistics by calendar year."""

    yearly = bt.copy()
    yearly["year"] = pd.to_datetime(yearly[date_col]).dt.year

    rows = []
    for year, group in yearly.groupby("year"):
        strategy_stats = performance_stats(group[strategy_col])
        benchmark_stats = performance_stats(group[benchmark_col])
        rows.append(
            {
                "year": year,
                "strategy_return": strategy_stats["total_return"],
                "benchmark_return": benchmark_stats["total_return"],
                "strategy_sharpe": strategy_stats["sharpe"],
                "benchmark_sharpe": benchmark_stats["sharpe"],
                "strategy_max_drawdown": strategy_stats["max_drawdown"],
                "benchmark_max_drawdown": benchmark_stats["max_drawdown"],
            }
        )

    return pd.DataFrame(rows)
