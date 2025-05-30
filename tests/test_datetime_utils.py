import numpy as np
import pandas as pd

from upgini.metadata import SearchKey
from upgini.utils.datetime_utils import (
    is_blocked_time_series,
    is_dates_distribution_valid,
    is_time_series,
)

pd.set_option("mode.chained_assignment", "raise")


def test_univariate_timeseries_detection():
    df = pd.DataFrame({"date": ["1990-01-01", "1991-01-01", "1992-01-01", "1993-01-01", "1994-01-01"]})
    assert is_time_series(df, "date")

    df = pd.DataFrame({"date": ["1990-01-01", "1990-01-01", "1992-01-01", "1993-01-01", "1994-01-01"]})
    assert not is_time_series(df, "date")

    df = pd.DataFrame({"date": ["1990-01-01", "1990-02-01", "1990-03-01", "1990-04-01", "1990-05-01"]})
    assert is_time_series(df, "date")

    df = pd.DataFrame({"date": ["1990-01-01", "1990-01-01", "1990-03-01", "1990-04-01", "1990-05-01"]})
    assert not is_time_series(df, "date")

    df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]})
    assert is_time_series(df, "date")

    df = pd.DataFrame({"date": ["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05"]})
    assert not is_time_series(df, "date")

    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01 00:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 02:00:00",
                "2020-01-01 03:00:00",
                "2020-01-01 04:00:00",
            ]
        }
    )
    assert is_time_series(df, "date")

    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:02",
                "2020-01-01 00:00:04",
                "2020-01-01 00:00:06",
                "2020-01-01 00:00:08",
            ]
        }
    )
    assert is_time_series(df, "date")


def test_multivariate_timeseries_detection():
    df = pd.DataFrame(
        {
            "date": [
                "1990-01-01",
                "1990-01-01",
                "1991-01-01",
                "1991-01-01",
                "1992-01-01",
                "1992-01-01",
                "1993-01-01",
                "1993-01-01",
                "1994-01-01",
                "1994-01-01",
            ]
        }
    )
    assert is_time_series(df, "date")

    df = pd.DataFrame(
        {
            "date": [
                "1990-01-01",
                "1990-01-01",
                "1992-01-01",
                "1992-01-01",
                "1993-01-01",
                "1993-01-01",
                "1994-01-01",
                "1994-01-01",
            ]
        }
    )
    assert not is_time_series(df, "date")

    df = pd.DataFrame({"date": ["1990-01-01", "1990-02-01", "1990-03-01", "1990-04-01", "1990-05-01"]})
    assert is_time_series(df, "date")

    df = pd.DataFrame({"date": ["1990-01-01", "1990-01-01", "1990-03-01", "1990-04-01", "1990-05-01"]})
    assert not is_time_series(df, "date")

    df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]})
    assert is_time_series(df, "date")

    df = pd.DataFrame({"date": ["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05"]})
    assert not is_time_series(df, "date")

    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01 00:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 02:00:00",
                "2020-01-01 03:00:00",
                "2020-01-01 04:00:00",
            ]
        }
    )
    assert is_time_series(df, "date")

    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:02",
                "2020-01-01 00:00:04",
                "2020-01-01 00:00:06",
                "2020-01-01 00:00:08",
            ]
        }
    )
    assert is_time_series(df, "date")


def test_multivariate_time_series():
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:02",
                "2020-01-01 00:00:04",
                "2020-01-01 00:00:06",
                "2020-01-01 00:00:08",
            ]
        }
    )
    assert not is_blocked_time_series(df, "date", ["date"])

    df = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-02-01")})
    assert not is_blocked_time_series(df, "date", ["date"])

    df = pd.DataFrame({"date": pd.date_range("2020-01-01", "2021-01-01")})
    assert not is_blocked_time_series(df, "date", ["date"])

    df1 = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2021-01-01"),
            "feature1": np.random.randint(0, 1000, 367),
            "feature2": np.random.randint(0, 1000, 367),
            "feature3": range(367),
        }
    )
    df2 = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2021-01-01"),
            "feature1": np.random.randint(0, 1000, 367),
            "feature2": np.random.randint(0, 1000, 367),
            "feature3": range(367),
        }
    )
    df = pd.concat([df1, df2])
    assert is_blocked_time_series(df, "date", ["date"])

    df1 = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2021-01-01"),
            "feature1": np.random.randint(0, 1000, 367),
            "feature2": np.random.randint(0, 1000, 367),
            "feature3": np.random.randint(0, 1000, 367),
        }
    )
    df2 = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2021-01-01"),
            "feature1": np.random.randint(0, 1000, 367),
            "feature2": np.random.randint(0, 1000, 367),
            "feature3": np.random.randint(0, 1000, 367),
        }
    )
    df = pd.concat([df1, df2])
    assert not is_blocked_time_series(df, "date", ["date"])

    assert is_blocked_time_series(df, "date", ["date", "feature3"])


def test_blocked_time_series_with_invalid_date():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", None, "invalid_date"],
            "feature1": np.random.randint(0, 1000, 4),
            "feature2": np.random.randint(0, 1000, 4),
            "feature3": np.random.randint(0, 1000, 4),
        }
    )
    assert not is_blocked_time_series(df, "date", ["date"])


def test_validate_dates_distribution():
    df = pd.DataFrame({"date": ["2020-01-01"] * 10 + ["2020-02-01"] * 20 + ["2020-03-01"] * 30 + ["2020-04-01"] * 40})
    is_valid = is_dates_distribution_valid(df, {})
    assert not is_valid

    df = pd.DataFrame({"date": ["2020-05-01"] * 10 + ["2020-02-01"] * 20 + ["2020-03-01"] * 30 + ["2020-04-01"] * 40})
    is_valid = is_dates_distribution_valid(df, {})
    assert is_valid

    df = pd.DataFrame(
        {
            "date2": ["2020-05-01"] * 10 + ["2020-02-01"] * 20 + ["2020-03-01"] * 30 + ["2020-04-01"] * 40,
            "date1": ["2020-01-01"] * 10 + ["2020-02-01"] * 20 + ["2020-03-01"] * 30 + ["2020-04-01"] * 40,
        }
    )
    is_valid = is_dates_distribution_valid(df, {"date1": SearchKey.DATE})
    assert not is_valid
