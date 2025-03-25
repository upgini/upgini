from upgini.utils.ts_utils import get_most_frequent_time_unit, trunc_datetime

import pandas as pd


def test_get_most_frequent_time_unit():
    def dates(str_values):
        return [pd.to_datetime(value) for value in str_values]

    # Test with single frequency (daily)
    df = pd.DataFrame({"id": [1, 1, 1], "date": dates(["2023-01-01", "2023-01-02", "2023-01-03"])})
    assert get_most_frequent_time_unit(df, ["id"], "date") == pd.Timedelta(days=1)

    # Test with irregular month intervals
    df = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "date": dates(["2023-01-01", "2023-03-01", "2023-07-01"]),
        }
    )
    assert get_most_frequent_time_unit(df, ["id"], "date") == pd.Timedelta(days=59)

    # Test with mixed frequencies, daily being most common
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "date": dates(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-03 12:00:00"]),
        }
    )
    assert get_most_frequent_time_unit(df, ["id"], "date") == pd.Timedelta(days=1)

    # Test with mixed frequencies, hourly being most common
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 1],
            "date": dates(
                ["2023-01-01 12:00:00", "2023-01-01 13:00:00", "2023-01-01 14:00:00", "2023-01-02", "2023-01-03"]
            ),
        }
    )
    assert get_most_frequent_time_unit(df, ["id"], "date") == pd.Timedelta(hours=1)

    # Test with multiple IDs
    df = pd.DataFrame(
        {
            "id1": [1, 1, 2, 2],
            "id2": ["A", "A", "B", "B"],
            "date": dates(["2023-01-01", "2023-01-02", "2023-01-01 12:00:00", "2023-01-01 13:00:00"]),
        }
    )
    assert get_most_frequent_time_unit(df, ["id1", "id2"], "date") == pd.Timedelta(hours=1)

    # Test with no IDs
    df = pd.DataFrame({"date": dates(["2023-01-01", "2023-01-02", "2023-01-03"])})
    assert get_most_frequent_time_unit(df, [], "date") == pd.Timedelta(days=1)

    # Test with empty dataframe
    df_empty = pd.DataFrame(columns=["id", "date"])
    assert get_most_frequent_time_unit(df_empty, ["id"], "date") is None


def test_trunc_datetime():
    # Test with single ID
    df = pd.DataFrame(
        {"id": [1, 1, 1, 1], "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])}
    )

    result = trunc_datetime(df, ["id"], "date", pd.DateOffset(days=2))
    assert len(result) == 2
    assert result["date"].min() == pd.to_datetime("2023-01-03")
    assert result["date"].max() == pd.to_datetime("2023-01-04")

    # Test with multiple IDs
    df = pd.DataFrame(
        {
            "id1": [1, 1, 2, 2],
            "id2": ["A", "A", "B", "B"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
        }
    )

    result = trunc_datetime(df, ["id1", "id2"], "date", pd.DateOffset(days=1))
    assert len(result) == 2
    assert result["date"].tolist() == [pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-04")]

    # Test with no IDs
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])})

    result = trunc_datetime(df, [], "date", pd.DateOffset(days=2))
    assert len(result) == 2
    assert result["date"].min() == pd.to_datetime("2023-01-03")
    assert result["date"].max() == pd.to_datetime("2023-01-04")
