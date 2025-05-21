import numpy as np
import pandas as pd
from typing import List
from upgini.utils.sample_utils import sample_time_series_trunc, sample_time_series, sample_time_series_train_eval
from upgini.utils.sample_utils import SampleColumns


def test_sample_time_series_trim_ids():
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
        }
    )

    # Test basic sampling with enough different IDs
    balanced_df = sample_time_series(
        df=df, id_columns=["id"], date_column="date", sample_size=6, min_different_ids_ratio=2 / 3
    )
    assert len(balanced_df) == 6
    assert balanced_df["id"].nunique() == 2


def test_sample_time_series_trim_dates():
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
            ],
        }
    )

    balanced_df = sample_time_series(
        df=df, id_columns=["id"], date_column="date", sample_size=4, min_different_ids_ratio=1.0
    )
    assert len(balanced_df) == 4
    assert balanced_df["id"].nunique() == 2
    assert len(balanced_df["date"].unique()) == 2


def test_balance_undersampling_time_series_multiple_ids():
    df = pd.DataFrame(
        {
            "id1": [1, 1, 1, 2, 2, 2],
            "id2": ["A", "A", "A", "B", "B", "B"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )

    balanced_df = sample_time_series(
        df=df, id_columns=["id1", "id2"], date_column="date", sample_size=4, min_different_ids_ratio=1.0
    )
    assert len(balanced_df) == 4
    assert balanced_df.groupby(["id1", "id2"]).ngroups == 2
    assert balanced_df.date.max() == "2020-01-03"


def test_sample_time_series_no_ids():
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
            ],
        }
    )
    balanced_df = sample_time_series(
        df=df, id_columns=[], date_column="date", sample_size=6, min_different_ids_ratio=2 / 3
    )
    assert len(balanced_df) == 6
    assert balanced_df.date.max() == "2020-01-09"
    assert balanced_df.date.min() == "2020-01-04"


def test_sample_time_series_shifted_dates():
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
            ],
        }
    )

    balanced_df = sample_time_series(
        df=df, id_columns=["id"], date_column="date", sample_size=6, min_different_ids_ratio=2 / 3
    )
    assert len(balanced_df) == 6
    assert balanced_df.groupby(["id"]).ngroups == 2
    assert balanced_df.date.max() == "2020-01-04"
    assert balanced_df.date.min() == "2020-01-02"


def test_sample_time_series_random_seed():
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "date": [
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
            ],
        }
    )

    balanced_df_1 = sample_time_series(
        df=df, id_columns=["id"], date_column="date", sample_size=6, min_different_ids_ratio=2 / 3, random_state=42
    )
    balanced_df_2 = sample_time_series(
        df=df, id_columns=["id"], date_column="date", sample_size=6, min_different_ids_ratio=2 / 3, random_state=24
    )

    # Different seeds should give different results while maintaining constraints
    assert not balanced_df_1.equals(balanced_df_2)
    assert len(balanced_df_1) == len(balanced_df_2) == 6
    assert balanced_df_1.groupby(["id"]).ngroups == balanced_df_2.groupby(["id"]).ngroups == 2
    assert balanced_df_1.date.max() == balanced_df_2.date.max() == "2020-01-04"
    assert balanced_df_1.date.min() == balanced_df_2.date.min() == "2020-01-02"


def test_sample_time_series_without_recent_dates():
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
            ],
        }
    )
    balanced_df_1 = sample_time_series(
        df=df,
        id_columns=["id"],
        date_column="date",
        sample_size=6,
        min_different_ids_ratio=2 / 3,
        random_state=42,
        prefer_recent_dates=False,
    )
    balanced_df_2 = sample_time_series(
        df=df,
        id_columns=["id"],
        date_column="date",
        sample_size=6,
        min_different_ids_ratio=2 / 3,
        random_state=24,
        prefer_recent_dates=False,
    )

    # Different seeds should give different results while maintaining constraints
    assert not balanced_df_1.equals(balanced_df_2)
    assert len(balanced_df_1) == len(balanced_df_2) == 6
    assert balanced_df_1.groupby(["id"]).ngroups == balanced_df_2.groupby(["id"]).ngroups == 2


def test_sample_time_series_trunc():
    def unique_dates(df: pd.DataFrame) -> List[str]:
        return pd.to_datetime(df["date"], unit="ms").dt.date.astype(str).unique().tolist()

    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "date": [
                pd.to_datetime(d).timestamp() * 1000
                for d in [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                ]
            ],
        }
    )

    # Test high frequency truncation
    sampled_df = sample_time_series_trunc(
        df=df,
        id_columns=["id"],
        date_column="date",
        sample_size=6,
        random_state=42,
        highfreq_trunc_lengths=[pd.DateOffset(days=2), pd.DateOffset(days=1)],
        lowfreq_trunc_lengths=[],
    )
    assert len(sampled_df) == 6
    assert sampled_df["id"].nunique() == 3
    assert unique_dates(sampled_df) == ["2020-01-02", "2020-01-03"]

    # Test highfreq truncation with second choice
    sampled_df = sample_time_series_trunc(
        df=df,
        id_columns=["id"],
        date_column="date",
        sample_size=5,
        random_state=42,
        highfreq_trunc_lengths=[pd.DateOffset(days=2), pd.DateOffset(days=1)],
        lowfreq_trunc_lengths=[],
    )
    assert len(sampled_df) == 3
    assert sampled_df["id"].nunique() == 3
    assert unique_dates(sampled_df) == ["2020-01-03"]

    # Test highfreq id truncation
    sampled_df = sample_time_series_trunc(
        df=df,
        id_columns=["id"],
        date_column="date",
        sample_size=2,
        random_state=42,
        highfreq_trunc_lengths=[pd.DateOffset(days=2), pd.DateOffset(days=1)],
        lowfreq_trunc_lengths=[],
    )
    assert len(sampled_df) == 2
    assert sampled_df["id"].nunique() == 2
    assert unique_dates(sampled_df) == ["2020-01-03"]

    # Test low frequency truncation
    df_lowfreq = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": [
                pd.to_datetime(d).timestamp() * 1000
                for d in ["2020-01-01", "2020-03-01", "2020-07-01", "2020-01-01", "2020-03-01", "2020-07-01"]
            ],
        }
    )

    sampled_df = sample_time_series_trunc(
        df=df_lowfreq,
        id_columns=["id"],
        date_column="date",
        sample_size=4,
        random_state=42,
        highfreq_trunc_lengths=[],
        lowfreq_trunc_lengths=[pd.DateOffset(months=5), pd.DateOffset(months=1)],
    )
    assert len(sampled_df) == 4
    assert sampled_df["id"].nunique() == 2
    assert unique_dates(sampled_df) == ["2020-03-01", "2020-07-01"]

    # Test truncation with second choice
    sampled_df = sample_time_series_trunc(
        df=df_lowfreq,
        id_columns=["id"],
        date_column="date",
        sample_size=2,
        random_state=42,
        highfreq_trunc_lengths=[],
        lowfreq_trunc_lengths=[pd.DateOffset(months=5), pd.DateOffset(months=1)],
    )
    assert len(sampled_df) == 2
    assert sampled_df["id"].nunique() == 2
    assert unique_dates(sampled_df) == ["2020-07-01"]

    # Test lowfreq id truncation
    sampled_df = sample_time_series_trunc(
        df=df_lowfreq,
        id_columns=["id"],
        date_column="date",
        sample_size=1,
        random_state=42,
        highfreq_trunc_lengths=[],
        lowfreq_trunc_lengths=[pd.DateOffset(months=5), pd.DateOffset(months=1)],
    )
    assert len(sampled_df) == 1
    assert sampled_df["id"].nunique() == 1
    assert unique_dates(sampled_df) == ["2020-07-01"]

    # Test empty id_columns
    sampled_df = sample_time_series_trunc(
        df=df,
        id_columns=None,
        date_column="date",
        sample_size=6,
        random_state=42,
        highfreq_trunc_lengths=[pd.DateOffset(days=2), pd.DateOffset(days=1)],
        lowfreq_trunc_lengths=[],
    )
    assert len(sampled_df) == 6
    assert sampled_df["id"].nunique() == 3
    assert unique_dates(sampled_df) == ["2020-01-02", "2020-01-03"]


def test_sample_time_series_train_eval():
    # Test basic functionality without eval set
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
        }
    )
    sample_columns = SampleColumns(date="date", target="target", ids=["id"])

    sampled_df = sample_time_series_train_eval(
        df=df, sample_columns=sample_columns, sample_size=6, trim_threshold=10, max_rows=8, random_state=42
    )
    assert len(sampled_df) == 6
    assert sampled_df["id"].nunique() == 2


def test_sample_time_series_train_eval_with_eval_set():
    # Test with eval set
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 1
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 2
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",  # id 2
            ],
            "eval_set_index": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # First 6 rows are train, last 6 are eval
        }
    )
    sample_columns = SampleColumns(date="date", target="target", ids=["id"], eval_set_index="eval_set_index")

    sampled_df = sample_time_series_train_eval(
        df=df, sample_columns=sample_columns, sample_size=4, trim_threshold=8, max_rows=9, random_state=42
    )
    assert len(sampled_df) == 9
    train_df = sampled_df[sampled_df["eval_set_index"] == 0]
    eval_df = sampled_df[sampled_df["eval_set_index"] == 1]
    assert len(train_df) == 3
    assert len(eval_df) == 6
    assert train_df["id"].nunique() == 1


def test_sample_time_series_train_eval_eval_set_trimming():
    # Test eval set trimming when it exceeds threshold
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 1
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 2
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 1
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 2
            ],
            "eval_set_index": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # First 6 rows are train, last 6 are eval
        }
    )
    sample_columns = SampleColumns(date="date", target="target", ids=["id"], eval_set_index="eval_set_index")

    sampled_df = sample_time_series_train_eval(
        df=df, sample_columns=sample_columns, sample_size=4, trim_threshold=8, max_rows=6, random_state=42
    )
    assert len(sampled_df) == 6
    train_df = sampled_df[sampled_df["eval_set_index"] == 0]
    eval_df = sampled_df[sampled_df["eval_set_index"] == 1]
    assert len(train_df) == 3
    assert len(eval_df) == 3
    assert train_df["id"].nunique() == 1
    assert eval_df["id"].nunique() == 1


def test_sample_time_series_train_eval_missing_ids():
    # Test handling of missing IDs in eval set
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4],
            "id2": [10, 10, 10, 20, 20, 20, 20, 20, 20, 40, 40, 40],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 1
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 2
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 2
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",  # id 4
            ],
            "eval_set_index": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # First 6 rows are train, last 6 are eval
        }
    )
    sample_columns = SampleColumns(date="date", target="target", ids=["id", "id2"], eval_set_index="eval_set_index")

    sampled_df = sample_time_series_train_eval(
        df=df,
        sample_columns=sample_columns,
        sample_size=3,  # This will force sampling of only one ID from train
        trim_threshold=8,
        max_rows=9,
        random_state=42,
    )
    assert len(sampled_df) == 6  # Only one ID remains in eval_set
    train_df = sampled_df[sampled_df["eval_set_index"] == 0]
    eval_df = sampled_df[sampled_df["eval_set_index"] == 1]
    assert len(train_df) == 3
    assert len(eval_df) == 3
    train_ids = {tuple(row) for row in train_df[["id", "id2"]].values}
    eval_ids = {tuple(row) for row in eval_df[["id", "id2"]].values}
    assert len(train_ids) == 1
    assert len(eval_ids) == 1
    assert train_ids == eval_ids
