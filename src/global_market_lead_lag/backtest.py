"""Backtesting and validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from global_market_lead_lag.config import StrategySpec
from global_market_lead_lag.metrics import performance_stats
from global_market_lead_lag.models import fit_ols_model, predict_with_model
from global_market_lead_lag.signals import build_signals


def train_test_backtest(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    date_col: str,
    mode: str = "long_only",
    threshold: float = 0.0,
    split: float = 0.70,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, object]:
    """Fit on the first part of the sample and backtest on the rest."""

    df = df.copy().sort_values(date_col)
    split_idx = int(len(df) * split)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:].copy()
    model = fit_ols_model(train, feature_col, target_col)
    bt = apply_strategy(model, test, feature_col, target_col, date_col, mode, threshold, cost_bps)
    return bt, model


def apply_strategy(
    model,
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    date_col: str,
    mode: str,
    threshold: float,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Apply a fitted model and signal rule to an out-of-sample frame."""

    out = df.copy().sort_values(date_col)
    out["predicted_return"] = predict_with_model(model, out, feature_col)
    out["signal"] = build_signals(out["predicted_return"], mode=mode, threshold=threshold)

    out["turnover"] = out["signal"].diff().abs()
    out.loc[out.index[0], "turnover"] = abs(out.loc[out.index[0], "signal"])

    out["cost"] = out["turnover"] * (cost_bps / 10000)
    out["strategy_return"] = out["signal"] * out[target_col] - out["cost"]
    out["buy_hold_return"] = out[target_col]
    return out


def evaluate_strategy_variants(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    date_col: str,
    strategy_specs: tuple[StrategySpec, ...],
    split: float = 0.70,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate several strategy variants on one train/test split."""

    backtests: dict[str, pd.DataFrame] = {}
    results = {}

    for spec in strategy_specs:
        bt, _ = train_test_backtest(
            df,
            feature_col=feature_col,
            target_col=target_col,
            date_col=date_col,
            mode=spec.mode,
            threshold=spec.threshold,
            split=split,
            cost_bps=cost_bps,
        )
        backtests[spec.name] = bt
        results[spec.name] = performance_stats(bt["strategy_return"])

    first_bt = next(iter(backtests.values()))
    results["Buy & Hold"] = performance_stats(first_bt["buy_hold_return"])
    return pd.DataFrame(results).T.sort_values("sharpe", ascending=False), backtests


def train_validation_test_split(
    df: pd.DataFrame,
    date_col: str = "japan_date",
    train_size: float = 0.60,
    validation_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series into train, validation, and test windows."""

    df = df.copy().sort_values(date_col)
    train_end = int(len(df) * train_size)
    validation_end = int(len(df) * (train_size + validation_size))

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()
    return train, validation, test


def select_on_validation(
    model,
    validation: pd.DataFrame,
    feature_col: str,
    target_col: str,
    date_col: str,
    strategy_specs: tuple[StrategySpec, ...],
    cost_bps: float = 5.0,
) -> tuple[StrategySpec, pd.DataFrame]:
    """Select the highest-Sharpe strategy on a validation window."""

    validation_results = {}
    for spec in strategy_specs:
        bt_val = apply_strategy(
            model,
            validation,
            feature_col=feature_col,
            target_col=target_col,
            date_col=date_col,
            mode=spec.mode,
            threshold=spec.threshold,
            cost_bps=cost_bps,
        )
        validation_results[spec.name] = performance_stats(bt_val["strategy_return"])

    table = pd.DataFrame(validation_results).T.sort_values("sharpe", ascending=False)
    selected_name = table.index[0]
    selected = next(spec for spec in strategy_specs if spec.name == selected_name)
    return selected, table


def run_train_validation_test(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    date_col: str,
    strategy_specs: tuple[StrategySpec, ...],
    cost_bps: float = 5.0,
) -> dict[str, object]:
    """Run one train/validation/test model-selection experiment."""

    train, validation, test = train_validation_test_split(df, date_col=date_col)
    selection_model = fit_ols_model(train, feature_col, target_col)
    selected, validation_results = select_on_validation(
        selection_model,
        validation,
        feature_col,
        target_col,
        date_col,
        strategy_specs,
        cost_bps,
    )

    live_model = fit_ols_model(pd.concat([train, validation], axis=0), feature_col, target_col)
    bt_test = apply_strategy(
        live_model,
        test,
        feature_col,
        target_col,
        date_col,
        selected.mode,
        selected.threshold,
        cost_bps,
    )

    test_results = pd.DataFrame(
        {
            f"Selected Strategy: {selected.name}": performance_stats(bt_test["strategy_return"]),
            "Buy & Hold": performance_stats(bt_test["buy_hold_return"]),
        }
    ).T

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "selected": selected,
        "validation_results": validation_results,
        "bt_test": bt_test,
        "test_results": test_results,
    }


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    date_col: str,
    strategy_specs: tuple[StrategySpec, ...],
    initial_train_size: int = 1500,
    validation_size: int = 250,
    test_size: int = 63,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window walk-forward strategy selection and testing."""

    df = df.copy().sort_values(date_col).reset_index(drop=True)
    fold_outputs = []
    fold_summaries = []

    test_start = initial_train_size + validation_size
    fold_id = 1

    while test_start < len(df):
        validation_start = test_start - validation_size
        test_end = min(test_start + test_size, len(df))

        train_fold = df.iloc[:validation_start].copy()
        validation_fold = df.iloc[validation_start:test_start].copy()
        test_fold = df.iloc[test_start:test_end].copy()

        if len(train_fold) < 30 or validation_fold.empty or test_fold.empty:
            break

        selection_model = fit_ols_model(train_fold, feature_col, target_col)
        selected, validation_table = select_on_validation(
            selection_model,
            validation_fold,
            feature_col,
            target_col,
            date_col,
            strategy_specs,
            cost_bps,
        )

        live_model = fit_ols_model(
            pd.concat([train_fold, validation_fold], axis=0),
            feature_col,
            target_col,
        )
        live = test_fold.copy().sort_values(date_col)
        live["predicted_return"] = predict_with_model(live_model, live, feature_col)
        live["signal"] = build_signals(
            live["predicted_return"],
            mode=selected.mode,
            threshold=selected.threshold,
        )
        live["fold"] = fold_id
        live["selected_strategy"] = selected.name
        live["selected_threshold"] = selected.threshold
        live["gross_strategy_return"] = live["signal"] * live[target_col]
        live["buy_hold_return"] = live[target_col]

        fold_outputs.append(live)
        fold_summaries.append(
            {
                "fold": fold_id,
                "train_start": train_fold[date_col].min(),
                "train_end": train_fold[date_col].max(),
                "validation_start": validation_fold[date_col].min(),
                "validation_end": validation_fold[date_col].max(),
                "test_start": test_fold[date_col].min(),
                "test_end": test_fold[date_col].max(),
                "selected_strategy": selected.name,
                "validation_sharpe": validation_table.loc[selected.name, "sharpe"],
                "validation_total_return": validation_table.loc[selected.name, "total_return"],
                "model_coefficient": live_model.params[feature_col],
                "model_r_squared_train_validation": live_model.rsquared,
                "test_rows": len(test_fold),
            }
        )

        test_start = test_end
        fold_id += 1

    if not fold_outputs:
        raise ValueError("No walk-forward folds were generated. Reduce the window sizes.")

    results = pd.concat(fold_outputs, axis=0).sort_values(date_col).reset_index(drop=True)
    results["turnover"] = results["signal"].diff().abs()
    results.loc[results.index[0], "turnover"] = abs(results.loc[results.index[0], "signal"])
    results["cost"] = results["turnover"] * (cost_bps / 10000)
    results["strategy_return"] = results["gross_strategy_return"] - results["cost"]

    return results, pd.DataFrame(fold_summaries)
