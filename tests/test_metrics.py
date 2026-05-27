import pandas as pd

from global_market_lead_lag.metrics import performance_stats, strategy_diagnostics


def test_performance_stats_returns_expected_fields():
    stats = performance_stats(pd.Series([0.01, -0.005, 0.002]))

    assert "annual_return" in stats
    assert "max_drawdown" in stats
    assert stats["win_rate"] == 2 / 3


def test_strategy_diagnostics_counts_exposure():
    bt = pd.DataFrame(
        {
            "signal": [0, 1, 1, -1],
            "turnover": [0, 1, 0, 2],
            "strategy_return": [0.0, 0.01, -0.005, 0.02],
        }
    )

    diagnostics = strategy_diagnostics(bt)

    assert diagnostics["days"] == 4
    assert diagnostics["invested_days"] == 3
    assert diagnostics["long_days"] == 2
    assert diagnostics["short_days"] == 1
