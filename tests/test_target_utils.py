import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from upgini.errors import ValidationError
from upgini.features_enricher import FeaturesEnricher
from upgini.metadata import SYSTEM_RECORD_ID, TARGET, ModelTaskType, SearchKey
from upgini.resource_bundle import bundle
from upgini.utils.target_utils import balance_undersample, define_task


def test_invalid_target():
    y = pd.Series(["", "", ""])
    with pytest.raises(ValidationError, match=bundle.get("empty_target")):
        define_task(y)

    y = pd.Series([np.nan, np.inf, -np.inf])
    with pytest.raises(ValidationError, match=bundle.get("empty_target")):
        define_task(y)

    y = pd.Series([1, 1, 1, 1, 1])
    with pytest.raises(ValidationError, match=bundle.get("dataset_constant_target")):
        define_task(y)


def test_define_binary_task_type():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    assert define_task(y, False) == ModelTaskType.BINARY
    assert define_task(y, True) == ModelTaskType.BINARY

    y = pd.Series(["a", "b", "a", "b", "a"])
    assert define_task(y, False) == ModelTaskType.BINARY
    assert define_task(y, True) == ModelTaskType.BINARY


def test_define_multiclass_task_type():
    y = pd.Series(range(1, 51))
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series([float(x) for x in range(1, 51)])
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series(range(0, 50))
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series(["a", "b", "c", "b", "a"])
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series(["0", "1", "2", "3", "a"])
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series([0.0, 3.0, 5.0, 0.0, 5.0, 0.0, 3.0, 0.0, 5.0, 0.0, 5.0, 0.0, 3.0, 0.0, 3.0, 5.0, 3.0])
    assert define_task(y, False) == ModelTaskType.MULTICLASS


def test_define_regression_task_type():
    y = pd.Series([0.0, 3.0, 5.0, 0.0, 5.0, 0.0, 3.0, 0.0, 5.0, 0.0, 5.0, 0.0, 3.0, 0.0, 3.0, 5.0, 3.0])
    assert define_task(y, True) == ModelTaskType.REGRESSION

    y = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5])
    assert define_task(y, False) == ModelTaskType.REGRESSION
    assert define_task(y, True) == ModelTaskType.REGRESSION

    y = pd.Series([0, 1, 2, 3, 4, 5, 6, 8])
    assert define_task(y, False) == ModelTaskType.REGRESSION
    assert define_task(y, True) == ModelTaskType.REGRESSION

    y = pd.Series([0.0, 3.0, 5.0, 0.0, 5.0, 0.0, 3.0])
    assert define_task(y, False) == ModelTaskType.REGRESSION
    assert define_task(y, True) == ModelTaskType.REGRESSION


def test_balance_undersampling_binary():
    df = pd.DataFrame({SYSTEM_RECORD_ID: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], TARGET: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]})
    balanced_df = balance_undersample(
        df, TARGET, ModelTaskType.BINARY, 42, imbalance_threshold=0.1, min_sample_threshold=2
    )
    # Get all minority class and 5x of majority class if minority class count (1)
    # more or equal to min_sample_threshold/2 (1)
    expected_df = pd.DataFrame({
        SYSTEM_RECORD_ID: [1, 2, 3, 7, 9, 10],
        TARGET: [0, 1, 0, 0, 0, 0]
    })
    assert_frame_equal(balanced_df.sort_values(by=SYSTEM_RECORD_ID).reset_index(drop=True), expected_df)

    balanced_df = balance_undersample(
        df, TARGET, ModelTaskType.BINARY, 42, imbalance_threshold=0.1, min_sample_threshold=8
    )
    # Get all minority class and fill up to min_sample_threshold (8) by majority class
    expected_df = pd.DataFrame({
        SYSTEM_RECORD_ID: [1, 2, 3, 4, 6, 7, 9, 10],
        TARGET: [0, 1, 0, 0, 0, 0, 0, 0]
    })
    assert_frame_equal(balanced_df.sort_values(by=SYSTEM_RECORD_ID).reset_index(drop=True), expected_df)

    df = pd.DataFrame({"system_record_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], TARGET: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]})
    balanced_df = balance_undersample(
        df, "target", ModelTaskType.BINARY, 42, imbalance_threshold=0.1, min_sample_threshold=4
    )
    # Get full dataset if majority class count (8) less than x5 of minority class count (2)
    assert_frame_equal(balanced_df, df)


def test_balance_undersaampling_multiclass():
    df = pd.DataFrame({
        SYSTEM_RECORD_ID: [1, 2, 3, 4, 5, 6],
        TARGET: ["a", "b", "c", "c", "b", "c"]
        # a - 1, b - 2, c - 3
    })
    balanced_df = balance_undersample(
        df, TARGET, ModelTaskType.MULTICLASS, 42, imbalance_threshold=0.1, min_sample_threshold=10
    )
    # Get full dataset if majority class count (3) less than x2 of 25% class (b) count (2)
    assert_frame_equal(balanced_df, df)

    df = pd.DataFrame({
        SYSTEM_RECORD_ID: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        TARGET: ["a", "b", "c", "c", "c", "b", "c", "d", "d", "d", "c"]
        # a - 1, b - 2, c - 5, d - 3
    })
    balanced_df = balance_undersample(
        df, TARGET, ModelTaskType.MULTICLASS, 42, imbalance_threshold=0.1, min_sample_threshold=10
    )
    expected_df = pd.DataFrame({
        SYSTEM_RECORD_ID: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        TARGET: ["a", "b", "c", "c", "c", "b", "d", "d", "d", "c"]
    })
    # Get all of 25% quantile class (b) and minor classes (a) and x2 (or all if less) of major classes
    assert_frame_equal(balanced_df.sort_values(by=SYSTEM_RECORD_ID).reset_index(drop=True), expected_df)


def test_binary_psi_calculation():
    df = pd.DataFrame({
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,   0, 0, 0, 0, 0, 1, 0, 1, 0, 1]
    })
    df["date"] = pd.date_range("2020-01-01", "2020-01-20")
    enricher = FeaturesEnricher(search_keys={"date": SearchKey.DATE}, logs_enabled=False)
    enricher._validate_PSI(df)
    assert not enricher.warning_counter.has_warnings()

    df = pd.DataFrame({
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,   0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    })
    df["date"] = pd.date_range("2020-01-01", "2020-01-20")
    enricher = FeaturesEnricher(search_keys={"date": SearchKey.DATE}, logs_enabled=False)
    enricher._validate_PSI(df)
    assert enricher.warning_counter._count == 1

    df = pd.DataFrame({
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,   0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        "eval_set_index": [0] * 10 + [1] * 10,
    })
    df["date"] = pd.date_range("2020-01-01", "2020-01-20")
    enricher = FeaturesEnricher(search_keys={"date": SearchKey.DATE}, logs_enabled=False)
    enricher._validate_PSI(df)
    assert enricher.warning_counter._count == 1

    df = pd.DataFrame({
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,   0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        "eval_set_index": [0] * 10 + [1] * 10,
    })
    df["date"] = pd.date_range("2020-01-01", "2020-01-20")
    enricher = FeaturesEnricher(search_keys={"date": SearchKey.DATE}, logs_enabled=False)
    enricher._validate_PSI(df)
    assert enricher.warning_counter._count == 2


def test_regression_psi_calculation():
    random = np.random.RandomState(42)
    df = pd.DataFrame({
        "target": random.rand(20)
    })
    df["date"] = pd.date_range("2020-01-01", "2020-01-20")
    enricher = FeaturesEnricher(search_keys={"date": SearchKey.DATE}, logs_enabled=False)
    enricher._validate_PSI(df)
    assert enricher.warning_counter._count == 1

    values1 = random.rand(10)
    values2 = values1.copy()
    values2[0] = 0.0
    values2[9] = 1.0
    df = pd.DataFrame({
        "target": list(values1) + list(values2)
    })
    df["date"] = pd.date_range("2020-01-01", "2020-01-20")
    enricher = FeaturesEnricher(search_keys={"date": SearchKey.DATE}, logs_enabled=False)
    enricher._validate_PSI(df)
    assert not enricher.warning_counter.has_warnings()
