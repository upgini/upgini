import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.date import (
    DatePercentile,
)
from upgini.autofe.utils import pydantic_parse_method


def test_date_percentile():
    data = pd.DataFrame(
        [
            ["2024-03-03", 2, None],
            ["2024-02-03", 2, None],
            ["2024-02-04", 34, None],
            ["2024-02-05", 32, None],
            ["2023-03-03", 60, None],
            ["2023-03-02", None, None],
        ],
        columns=["date", "feature", "feature2"],
    )
    operand = DatePercentile(
        zero_month=2,
        zero_year=2024,
        zero_bounds="[0.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.6, 6.2, 6.8, 7.3999999999999995, 8.0, 8.6, 9.2, "
        "9.8, 10.4, 11.0, 11.6, 12.200000000000001, 12.799999999999999, 13.4, 14.0, 14.6, 15.2, 15.8, 16.4, 17.0,"
        " 17.6, 18.200000000000003, 18.8, 19.4, 20.0, 20.6, 21.200000000000003, 21.8, 22.400000000000002, 23.0, 23.6,"
        " 24.2, 24.8, 25.4, 26.0, 26.599999999999998, 27.2, 27.8, 28.4, 29.0, 29.6, 30.2, 30.799999999999997, 31.4,"
        " 32.0, 32.04, 32.08, 32.12, 32.16, 32.2, 32.24, 32.28, 32.32, 32.36, 32.4, 32.44, 32.48, 32.52, 32.56, 32.6, "
        "32.64, 32.68, 32.72, 32.76, 32.8, 32.84, 32.88, 32.92, 32.96, 33.0, 33.04, 33.08, 33.12, 33.16, 33.2, 33.24,"
        " 33.28, 33.32, 33.36, 33.4, 33.44, 33.48, 33.52, 33.56, 33.6, 33.64, 33.68, 33.72, 33.76, 33.8, 33.84, 33.88,"
        " 33.92, 33.96]",
    )

    expected_values = pd.Series([None, 1, 100, 51, 100, None])
    assert_series_equal(operand.calculate(left=data.date, right=data.feature), expected_values)
    assert_series_equal(
        operand.calculate(left=data.date, right=data.feature2), pd.Series([None] * len(data), dtype=float)
    )


def test_date_percentile_parse_obj():
    date_percentile = DatePercentile(
        zero_month=3,
        zero_year=2023,
        zero_bounds=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    )

    date_percentile_dict = date_percentile.get_params()
    parsed_date_percentile = pydantic_parse_method(DatePercentile)(date_percentile_dict)

    assert parsed_date_percentile.zero_month == 3
    assert parsed_date_percentile.zero_year == 2023
    assert parsed_date_percentile.zero_bounds == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
