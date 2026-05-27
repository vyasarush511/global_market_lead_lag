import numpy as np
import pandas as pd

from global_market_lead_lag.backtest import train_validation_test_split


def test_train_validation_test_split_preserves_time_order():
    df = pd.DataFrame(
        {
            "japan_date": pd.date_range("2020-01-01", periods=100),
            "feature": np.arange(100),
            "target": np.arange(100),
        }
    )

    train, validation, test = train_validation_test_split(df, train_size=0.6, validation_size=0.2)

    assert len(train) == 60
    assert len(validation) == 20
    assert len(test) == 20
    assert train["japan_date"].max() < validation["japan_date"].min()
    assert validation["japan_date"].max() < test["japan_date"].min()
