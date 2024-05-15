from datetime import datetime

import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.date import DateDiff, DateDiffType2, DateListDiff, DateListDiffBounded, DatePercentile


def test_date_diff():
    df = pd.DataFrame(
        [
            ["2022-10-10", pd.to_datetime("1993-12-10").timestamp()],
            ["2022-10-10", pd.to_datetime("2023-10-10").timestamp()],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiff(right_unit="s")
    expected_result = pd.Series([10531, None])
    assert_series_equal(operand.calculate_binary(df.date1, df.date2), expected_result)


def test_date_diff_type2():
    df = pd.DataFrame(
        [
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 12, 10)],
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 4, 10)],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiffType2(left_unit="s")
    expected_result = pd.Series([61.0, 182.0])
    actual = operand.calculate_binary(df.date1, df.date2)
    assert_series_equal(actual, expected_result)


def test_date_diff_list():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["1993-12-10", "1993-12-11"]],
            ["2022-10-10", ["1993-12-10", "1993-12-10"]],
            ["2022-10-10", ["2023-10-10"]],
            ["2022-10-10", []],
        ],
        columns=["date1", "date2"],
    )

    def check(aggregation, expected_name, expected_values):
        operand = DateListDiff(aggregation=aggregation)
        assert operand.name == expected_name
        assert_series_equal(operand.calculate_binary(df.date1, df.date2).rename(None), expected_values)

    check(aggregation="min", expected_name="date_diff_min", expected_values=pd.Series([10530, 10531, None, None]))
    check(aggregation="max", expected_name="date_diff_max", expected_values=pd.Series([10531, 10531, None, None]))
    check(aggregation="mean", expected_name="date_diff_mean", expected_values=pd.Series([10530.5, 10531, None, None]))
    check(aggregation="nunique", expected_name="date_diff_nunique", expected_values=pd.Series([2, 1, 0, 0]))


def test_date_diff_list_bounded():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["2013-12-10", "2013-12-11", "1999-12-11"]],
            [
                "2022-10-10",
                [
                    "2013-12-10",
                    "2003-12-11",
                    "1999-12-11",
                    "1993-12-11",
                    "1983-12-11",
                    "1973-12-11",
                    "1959-12-11",
                ],
            ],
            ["2022-10-10", ["2003-12-10", "2003-12-10"]],
            ["2022-10-10", ["2023-10-10", "1993-12-10"]],
            ["2022-10-10", []],
        ],
        columns=["date1", "date2"],
    )

    def check_num_by_years(lower_bound, upper_bound, expected_name, expected_values):
        operand = DateListDiffBounded(
            diff_unit="Y", aggregation="count", lower_bound=lower_bound, upper_bound=upper_bound
        )
        assert operand.name == expected_name
        assert_series_equal(operand.calculate_binary(df.date1, df.date2).rename(None), expected_values)

    check_num_by_years(0, 18, "date_diff_Y_0_18_count", pd.Series([2, 1, 0, 0, 0]))
    check_num_by_years(18, 23, "date_diff_Y_18_23_count", pd.Series([1, 2, 2, 0, 0]))
    check_num_by_years(23, 30, "date_diff_Y_23_30_count", pd.Series([0, 1, 0, 1, 0]))
    check_num_by_years(30, 45, "date_diff_Y_30_45_count", pd.Series([0, 1, 0, 0, 0]))
    check_num_by_years(45, 60, "date_diff_Y_45_60_count", pd.Series([0, 1, 0, 0, 0]))
    check_num_by_years(60, None, "date_diff_Y_60_plusinf_count", pd.Series([0, 1, 0, 0, 0]))


def test_date_percentile():
    data = pd.DataFrame(
        [
            ["2024-03-03", 2],
            ["2024-02-03", 2],
            ["2024-02-04", 34],
            ["2024-02-05", 32],
            ["2023-03-03", 60],
            ["2023-03-02", None],
        ],
        columns=["date", "feature"],
    )
    operand = DatePercentile(
        zero_month=2,
        zero_year=2024,
        zero_bounds="[0.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.6, 6.2, 6.8, 7.3999999999999995, 8.0, 8.6, 9.2, 9.8, 10.4, 11.0, 11.6, 12.200000000000001, 12.799999999999999, 13.4, 14.0, 14.6, 15.2, 15.8, 16.4, 17.0, 17.6, 18.200000000000003, 18.8, 19.4, 20.0, 20.6, 21.200000000000003, 21.8, 22.400000000000002, 23.0, 23.6, 24.2, 24.8, 25.4, 26.0, 26.599999999999998, 27.2, 27.8, 28.4, 29.0, 29.6, 30.2, 30.799999999999997, 31.4, 32.0, 32.04, 32.08, 32.12, 32.16, 32.2, 32.24, 32.28, 32.32, 32.36, 32.4, 32.44, 32.48, 32.52, 32.56, 32.6, 32.64, 32.68, 32.72, 32.76, 32.8, 32.84, 32.88, 32.92, 32.96, 33.0, 33.04, 33.08, 33.12, 33.16, 33.2, 33.24, 33.28, 33.32, 33.36, 33.4, 33.44, 33.48, 33.52, 33.56, 33.6, 33.64, 33.68, 33.72, 33.76, 33.8, 33.84, 33.88, 33.92, 33.96]",
    )

    expected_values = pd.Series([None, 1, 100, 51, 100, None])
    assert_series_equal(operand.calculate(left=data.date, right=data.feature), expected_values)
