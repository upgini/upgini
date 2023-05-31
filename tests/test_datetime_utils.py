import pandas as pd

from upgini.utils.datetime_utils import is_time_series

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
