from datetime import datetime

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.date import (
    DateDiff,
    DateDiffType2,
    DateListDiff,
    DateListDiffAggWithinBounds,
    DateListDiffBounded,
    DateListDiffLists,
)
from upgini.autofe.utils import pydantic_parse_method


def test_date_diff():
    df = pd.DataFrame(
        [
            ["2022-10-10", pd.to_datetime("1993-12-10").timestamp(), None],
            ["2022-10-10", pd.to_datetime("2023-10-10").timestamp(), None],
            ["2022-10-10", pd.to_datetime("1966-10-10").timestamp(), None],
            ["1022-10-10", pd.to_datetime("1966-10-10").timestamp(), None],
            [None, pd.to_datetime("1966-10-10").timestamp(), None],
            ["2022-10-10", None, None],
            [None, None, None],
        ],
        columns=["date1", "date2", "date3"],
    )

    operator = DateDiff(right_unit="s")
    expected_result = pd.Series([10531, -365.0, 20454, None, None, None, None], dtype=float)
    assert_series_equal(operator.calculate(left=df.date1, right=df.date2).as_series(), expected_result)

    operator = DateDiff(right_unit="s", replace_negative=True)
    expected_result = pd.Series([10531, None, 20454, None, None, None, None], dtype=float)
    assert_series_equal(operator.calculate(left=df.date1, right=df.date2).as_series(), expected_result)

    operator = DateDiff(right_unit="s")
    expected_result = pd.Series([None, None, None, None, None, None, None], dtype=float)
    assert_series_equal(operator.calculate(left=df.date1, right=df.date3).as_series(), expected_result)


def test_date_diff_type2():
    df = pd.DataFrame(
        [
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 12, 10), None],
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 4, 10), None],
            [pd.to_datetime("2022-10-10").timestamp(), datetime(993, 4, 10), None],
            [None, datetime(1993, 4, 10), None],
            [pd.to_datetime("2022-10-10").timestamp(), None, None],
            [None, None, None],
        ],
        columns=["date1", "date2", "date3"],
    )

    operator = DateDiffType2(left_unit="s")
    expected_result = pd.Series([61.0, 182.0, None, None, None, None])
    actual = operator.calculate(left=df.date1, right=df.date2).as_series()
    assert_series_equal(actual, expected_result)

    expected_result = pd.Series([None, None, None, None, None, None], dtype=float)
    actual = operator.calculate(left=df.date1, right=df.date3).as_series()
    assert_series_equal(actual, expected_result)


def test_date_diff_list():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["1993-12-10", "1993-12-11"], None],
            ["2022-10-10", ["1993-12-10", "1993-12-10"], None],
            ["2022-10-10", ["2023-10-10"], None],
            ["2022-10-10", ["1023-10-10"], None],
            ["2022-10-10", [], None],
        ],
        columns=["date1", "date2", "date3"],
    )

    def check(aggregation, expected_formula, expected_values):
        operator = DateListDiff(aggregation=aggregation)
        assert operator.to_formula() == expected_formula
        assert_series_equal(operator.calculate(left=df.date1, right=df.date2).as_series().rename(None), expected_values)

    check(
        aggregation="min",
        expected_formula="date_diff_min",
        expected_values=pd.Series([10530, 10531, -365.0, None, None]),
    )
    check(
        aggregation="max",
        expected_formula="date_diff_max",
        expected_values=pd.Series([10531, 10531, -365.0, None, None]),
    )
    check(
        aggregation="mean",
        expected_formula="date_diff_mean",
        expected_values=pd.Series([10530.5, 10531, -365.0, None, None]),
    )
    check(
        aggregation="nunique",
        expected_formula="date_diff_nunique",
        expected_values=pd.Series([2.0, 1.0, 1.0, 1.0, 0.0]),
    )

    operator = DateListDiff(aggregation="mean")
    df1 = df.loc[[2], :]
    assert_series_equal(
        operator.calculate(left=df1.date1, right=df1.date2).as_series().rename(None).reset_index(drop=True),
        pd.Series([-365.0]),
    )

    operator = DateListDiff(aggregation="nunique")
    df1 = df.loc[len(df) - 1 :, :]
    assert_series_equal(
        operator.calculate(left=df1.date1, right=df1.date2).as_series().rename(None).reset_index(drop=True),
        pd.Series([0.0]),
    )

    operator = DateListDiff(aggregation="min", replace_negative=True)
    assert_series_equal(
        operator.calculate(left=df.date1, right=df.date2).as_series().rename(None),
        pd.Series([10530, 10531, None, None, None]),
    )

    operator = DateListDiff(aggregation="min")
    assert_series_equal(
        operator.calculate(left=df.date1, right=df.date3).as_series().rename(None),
        pd.Series([None, None, None, None, None], dtype=float),
    )


def test_date_diff_list_bounded():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["2013-12-10", "2013-12-11", "1999-12-11"], None],
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
                None,
            ],
            ["2022-10-10", ["2003-12-10", "2003-12-10"], None],
            ["2022-10-10", ["2023-10-10", "1993-12-10"], None],
            ["1022-10-10", ["2023-10-10", "1993-12-10"], None],
            ["2022-10-10", [], None],
        ],
        columns=["date1", "date2", "date3"],
    )

    def check_num_by_years(lower_bound, upper_bound, expected_formula, expected_values, normalize=False):
        operator = DateListDiffBounded(
            diff_unit="Y", aggregation="count", lower_bound=lower_bound, upper_bound=upper_bound, normalize=normalize
        )
        assert operator.to_formula() == expected_formula
        assert_series_equal(operator.calculate(left=df.date1, right=df.date2).as_series().rename(None), expected_values)

    check_num_by_years(0, 18, "date_diff_Y_0_18_count", pd.Series([2.0, 1.0, 0.0, 0.0, None, 0.0]))
    check_num_by_years(18, 23, "date_diff_Y_18_23_count", pd.Series([1.0, 2.0, 2.0, 0.0, None, 0.0]))
    check_num_by_years(23, 30, "date_diff_Y_23_30_count", pd.Series([0.0, 1.0, 0.0, 1.0, None, 0.0]))
    check_num_by_years(30, 45, "date_diff_Y_30_45_count", pd.Series([0.0, 1.0, 0.0, 0.0, None, 0.0]))
    check_num_by_years(45, 60, "date_diff_Y_45_60_count", pd.Series([0.0, 1.0, 0.0, 0.0, None, 0.0]))
    check_num_by_years(60, None, "date_diff_Y_60_plusinf_count", pd.Series([0.0, 1.0, 0.0, 0.0, None, 0.0]))

    # Test with normalization
    check_num_by_years(
        0, 18, "date_diff_Y_0_18_count_norm", pd.Series([2.0 / 3.0, 1.0 / 7.0, 0.0, 0.0, None, 0.0]), normalize=True
    )
    check_num_by_years(
        18,
        23,
        "date_diff_Y_18_23_count_norm",
        pd.Series([1.0 / 3.0, 2.0 / 7.0, 2.0 / 2.0, 0.0, None, 0.0]),
        normalize=True,
    )

    operator = DateListDiffBounded(diff_unit="Y", aggregation="count", lower_bound=0, upper_bound=18)
    assert_series_equal(
        operator.calculate(left=df.date1, right=df.date3).as_series().rename(None),
        pd.Series([None] * len(df), dtype=float),
    )


def test_date_list_diff_bounded_from_formula():
    # Test with both bounds specified
    op = DateListDiffBounded.from_formula("date_diff_Y_18_23_count")
    assert op.diff_unit == "Y"
    assert op.lower_bound == 18
    assert op.upper_bound == 23
    assert op.aggregation == "count"
    assert op.normalize is False
    assert op.to_formula() == "date_diff_Y_18_23_count"

    # Test with only lower bound
    op = DateListDiffBounded.from_formula("date_diff_D_60_plusinf_mean")
    assert op.diff_unit == "D"
    assert op.lower_bound == 60
    assert op.upper_bound is None
    assert op.aggregation == "mean"
    assert op.normalize is False
    assert op.to_formula() == "date_diff_D_60_plusinf_mean"

    # Test with only upper bound
    op = DateListDiffBounded.from_formula("date_diff_Y_minusinf_18_nunique")
    assert op.diff_unit == "Y"
    assert op.lower_bound is None
    assert op.upper_bound == 18
    assert op.aggregation == "nunique"
    assert op.normalize is False
    assert op.to_formula() == "date_diff_Y_minusinf_18_nunique"

    # Test with normalization
    op = DateListDiffBounded.from_formula("date_diff_Y_18_23_count_norm")
    assert op.diff_unit == "Y"
    assert op.lower_bound == 18
    assert op.upper_bound == 23
    assert op.aggregation == "count"
    assert op.normalize is True
    assert op.to_formula() == "date_diff_Y_18_23_count_norm"

    # Test invalid formula returns None
    assert DateListDiffBounded.from_formula("invalid_formula") is None
    assert DateListDiffBounded.from_formula("date_diff_invalid") is None


def test_date_list_diff_bounded_parse_obj():
    date_diff = DateListDiffBounded(
        diff_unit="Y", lower_bound=18, upper_bound=25, aggregation="count", replace_negative=True
    )

    date_diff_dict = date_diff.get_params()
    parsed_date_diff = pydantic_parse_method(DateListDiffBounded)(date_diff_dict)

    assert parsed_date_diff.diff_unit == "Y"
    assert parsed_date_diff.lower_bound == 18
    assert parsed_date_diff.upper_bound == 25
    assert parsed_date_diff.aggregation == "count"
    assert parsed_date_diff.replace_negative is True
    assert parsed_date_diff.normalize is None
    assert parsed_date_diff.to_formula() == "date_diff_Y_18_25_count"

    # Test with normalization
    date_diff = DateListDiffBounded(diff_unit="Y", lower_bound=18, upper_bound=25, aggregation="count", normalize=True)

    date_diff_dict = date_diff.get_params()
    parsed_date_diff = pydantic_parse_method(DateListDiffBounded)(date_diff_dict)

    assert parsed_date_diff.diff_unit == "Y"
    assert parsed_date_diff.lower_bound == 18
    assert parsed_date_diff.upper_bound == 25
    assert parsed_date_diff.aggregation == "count"
    assert parsed_date_diff.normalize is True
    assert parsed_date_diff.to_formula() == "date_diff_Y_18_25_count_norm"


def test_date_list_diff_lists():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["1993-12-10", "1993-12-11"], None],
            ["2022-10-10", ["1993-12-10", "1993-12-10"], None],
            ["2022-10-10", ["2023-10-10"], None],
            ["2022-10-10", [], None],
        ],
        columns=["date1", "date2", "date3"],
    )

    operator = DateListDiffLists()
    diff_lists = operator.calculate(left=df.date1, right=df.date2).as_series()
    assert diff_lists.iloc[0] == [10531.0, 10530.0]
    assert diff_lists.iloc[1] == [10531.0, 10531.0]
    assert diff_lists.iloc[2] == [-365.0]
    assert diff_lists.iloc[3] == []

    operator = DateListDiffLists(replace_negative=True)
    diff_lists = operator.calculate(left=df.date1, right=df.date2).as_series()
    assert diff_lists.iloc[2] == []


def test_date_list_diff_agg_within_bounds():
    diff_lists = pd.Series(
        [
            np.array([8.0, 20.0, 25.0]),
            np.array([19.0, 22.0]),
            np.array([]),
            None,
        ]
    )

    operator = DateListDiffAggWithinBounds(lower_bound=18, upper_bound=23, aggregation="count")
    assert_series_equal(
        operator.calculate(data=diff_lists).as_series(),
        pd.Series([1.0, 2.0, 0.0, np.nan]),
    )

    operator = DateListDiffAggWithinBounds(lower_bound=0, upper_bound=18, aggregation="count", normalize=True)
    assert_series_equal(
        operator.calculate(data=diff_lists).as_series(),
        pd.Series([1.0 / 3.0, 0.0, 0.0, np.nan]),
    )

    operator = DateListDiffAggWithinBounds(lower_bound=None, upper_bound=None, aggregation="nunique", normalize=False)
    assert_series_equal(
        operator.calculate(data=diff_lists).as_series(),
        pd.Series([3.0, 2.0, 0.0, np.nan]),
    )


def test_date_list_diff_bounded_composition():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["2013-12-10", "2013-12-11", "1999-12-11"], None],
            ["2022-10-10", ["2023-10-10", "1993-12-10"], None],
        ],
        columns=["date1", "date2", "date3"],
    )

    lists_op = DateListDiffLists(diff_unit="Y")
    agg_op = DateListDiffAggWithinBounds(lower_bound=0, upper_bound=18, aggregation="count")
    bounded = DateListDiffBounded(diff_unit="Y", lower_bound=0, upper_bound=18, aggregation="count")
    composed = agg_op.calculate(data=lists_op.calculate(left=df.date1, right=df.date2))
    assert_series_equal(
        composed.as_series(),
        bounded.calculate(left=df.date1, right=df.date2).as_series(),
    )
