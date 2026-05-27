import pandas as pd

from global_market_lead_lag.signals import build_signals


def test_threshold_long_short_uses_cash_between_thresholds():
    predicted = pd.Series([-0.002, -0.0002, 0.0, 0.0002, 0.002])

    signals = build_signals(predicted, mode="threshold_long_short", threshold=0.001)

    assert signals.tolist() == [-1, 0, 0, 0, 1]
