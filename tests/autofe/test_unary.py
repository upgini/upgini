import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from upgini.autofe.feature import Feature


def test_abs():
    f = Feature.from_formula("abs(f2)")
    df = pd.DataFrame(
        {
            "f2": [None, None, None, None, None, None],
        }
    )
    result = f.calculate(df)
    print(result)
    expected = pd.DataFrame({"f2": [None, None, None, None, None, None]})
    assert_series_equal(result, expected["f2"].astype(np.float64))
