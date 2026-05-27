"""Data download and feature construction."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_market_data(
    tickers: dict[str, str],
    start_date: str,
    end_date: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLC data from Yahoo Finance."""

    raw = yf.download(
        list(tickers.values()),
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if raw.empty:
        raise ValueError("Downloaded market data is empty.")

    return raw


def extract_open_close(
    raw: pd.DataFrame,
    tickers: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and rename open/close price panels from yfinance output."""

    reverse_tickers = {ticker: name for name, ticker in tickers.items()}

    open_prices = raw["Open"].rename(columns=reverse_tickers)
    close_prices = raw["Close"].rename(columns=reverse_tickers)

    expected = set(tickers)
    missing_open = expected.difference(open_prices.columns)
    missing_close = expected.difference(close_prices.columns)

    if missing_open or missing_close:
        raise ValueError(
            "Missing expected ticker columns. "
            f"Open missing={sorted(missing_open)}, close missing={sorted(missing_close)}"
        )

    return open_prices, close_prices


def build_open_to_close_returns(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate regular-session open-to-close returns."""

    return (close_prices / open_prices - 1).dropna(how="all")


def build_close_to_close_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate close-to-close returns."""

    return close_prices.pct_change().dropna(how="all")


def _series_to_frame(series: pd.Series, date_col: str, value_col: str) -> pd.DataFrame:
    frame = series.dropna().reset_index()
    frame.columns = [date_col, value_col]
    return frame


def build_japan_us_tradable_dataset(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build the main no-lookahead U.S.-to-Japan session dataset.

    The feature is the S&P 500 open-to-close return from the latest completed
    U.S. session. The target is Nikkei 225 open-to-close return in the next
    Japanese session.
    """

    session_returns = build_open_to_close_returns(open_prices, close_prices)

    us_signal = _series_to_frame(
        session_returns["sp500"],
        "us_date",
        "feature_sp500_open_to_close",
    )
    jp_target = _series_to_frame(
        session_returns["nikkei"],
        "japan_date",
        "target_nikkei_open_to_close",
    )

    return pd.merge_asof(
        jp_target.sort_values("japan_date"),
        us_signal.sort_values("us_date"),
        left_on="japan_date",
        right_on="us_date",
        direction="backward",
        allow_exact_matches=False,
    ).dropna()


def build_directional_close_to_close_dataset(
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact lead-lag table for exploratory correlation analysis."""

    returns = build_close_to_close_returns(close_prices)[["nikkei", "sp500"]].dropna()
    lead_lag = returns.copy()
    lead_lag["nikkei_lag1"] = lead_lag["nikkei"].shift(1)
    lead_lag["sp500_lag1"] = lead_lag["sp500"].shift(1)
    return lead_lag.dropna()
