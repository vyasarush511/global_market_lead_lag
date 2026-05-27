"""Signal construction rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_signals(
    predicted_return: pd.Series,
    mode: str = "long_only",
    threshold: float = 0.0,
) -> np.ndarray:
    """Convert model-predicted returns into trade signals.

    Signals use 1 for long, -1 for short, and 0 for cash.
    """

    if mode == "long_only":
        return np.where(predicted_return > threshold, 1, 0)

    if mode == "long_short":
        return np.where(predicted_return > threshold, 1, -1)

    if mode == "threshold_long_only":
        return np.where(predicted_return > threshold, 1, 0)

    if mode == "threshold_long_short":
        return np.select(
            [predicted_return > threshold, predicted_return < -threshold],
            [1, -1],
            default=0,
        )

    raise ValueError(f"Unknown signal mode: {mode}")
