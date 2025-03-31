import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.timeseries.volatility import VolatilityBase


def test_volatility_base_get_returns():
    df = pd.Series(
        [100, 110, 99, 121, np.nan],
        index=pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-05", "2024-05-06"]),
        name="value",
    )

    returns = VolatilityBase._get_returns(df, "1D")

    expected_returns = pd.Series([0.0, 0.1, -0.1, 0.0, 0.0], index=df.index, name="value")

    assert_series_equal(returns, expected_returns)
