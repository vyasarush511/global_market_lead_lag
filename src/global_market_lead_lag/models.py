"""Statistical models used in the lead-lag study."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def fit_ols_model(train: pd.DataFrame, feature_col: str, target_col: str):
    """Fit a one-factor OLS model."""

    x_train = sm.add_constant(train[feature_col], has_constant="add")
    y_train = train[target_col]
    return sm.OLS(y_train, x_train).fit()


def regression_summary(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
) -> pd.Series:
    """Return the key one-factor regression statistics."""

    model = fit_ols_model(df, feature_col, target_col)
    return pd.Series(
        {
            "coefficient": model.params[feature_col],
            "p_value": model.pvalues[feature_col],
            "r_squared": model.rsquared,
            "observations": int(model.nobs),
        }
    )


def predict_with_model(model, df: pd.DataFrame, feature_col: str) -> pd.Series:
    """Predict returns from a fitted statsmodels model."""

    x = sm.add_constant(df[feature_col], has_constant="add")
    return model.predict(x)
