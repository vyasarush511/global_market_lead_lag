"""Project configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarketConfig:
    """Default data and evaluation settings for the research pipeline."""

    tickers: dict[str, str] = field(
        default_factory=lambda: {
            "nikkei": "^N225",
            "sp500": "^GSPC",
        }
    )
    start_date: str = "2010-01-01"
    end_date: str | None = None
    train_split: float = 0.70
    base_cost_bps: float = 5.0
    periods_per_year: int = 252


@dataclass(frozen=True)
class StrategySpec:
    """A named trading rule applied to model-predicted returns."""

    name: str
    mode: str
    threshold: float


DEFAULT_STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    StrategySpec("Long Only", "long_only", 0.0),
    StrategySpec("Long Short", "long_short", 0.0),
    StrategySpec("Threshold Long Only 5bps", "threshold_long_only", 0.0005),
    StrategySpec("Threshold Long Short 5bps", "threshold_long_short", 0.0005),
    StrategySpec("Threshold Long Only 10bps", "threshold_long_only", 0.0010),
    StrategySpec("Threshold Long Short 10bps", "threshold_long_short", 0.0010),
)
