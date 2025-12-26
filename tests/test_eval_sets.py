import numpy as np
import pandas as pd
from requests_mock.mocker import Mocker

from upgini.dataset import Dataset
from upgini.features_enricher import FeaturesEnricher
from upgini.http import _RestClient
from upgini.metadata import EVAL_SET_INDEX, TARGET, SearchKey

from .test_features_enricher import DataFrameWrapper, TestException
from .utils import mock_default_requests


def test_eval_sets_balanced_binary_classification(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    # Create balanced binary classification data
    np.random.seed(42)
    train_size = 1000
    eval_size = 200

    train_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size)],
            "feature1": np.random.randn(train_size),
        }
    )
    train_y = pd.Series(np.random.choice([0, 1], size=train_size, p=[0.5, 0.5]))

    eval_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size, train_size + eval_size)],
            "feature1": np.random.randn(eval_size),
        }
    )
    eval_y = pd.Series(np.random.choice([0, 1], size=eval_size, p=[0.5, 0.5]))

    result_wrapper = DataFrameWrapper()

    def mocked_initial_search(self, trace_id, file_path, metadata, metrics, search_customization):
        result_wrapper.df = pd.read_parquet(file_path)
        raise TestException

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    old_min_rows_count = Dataset.MIN_ROWS_COUNT
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher = FeaturesEnricher(
            search_keys={"phone": SearchKey.PHONE},
            endpoint=url,
            api_key="fake_api_key",
            logs_enabled=False,
        )

        try:
            enricher.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], calculate_metrics=False)
        except TestException:
            pass

        # Verify eval set is included in uploaded dataframe
        uploaded_df = result_wrapper.df
        assert uploaded_df is not None
        assert EVAL_SET_INDEX in uploaded_df.columns

        # Check train set (index 0)
        train_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 0]
        assert len(train_rows) > 0, "Train set should be included"
        assert len(train_rows) <= train_size, "Train set may be downsampled but should exist"

        # Check eval set (index 1)
        eval_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 1]
        assert len(eval_rows) > 0, "Eval set should be included"
        assert len(eval_rows) <= eval_size, "Eval set may be downsampled but should exist"
        assert not eval_rows[TARGET].isna().all(), "Eval set should have targets"

    finally:
        _RestClient.initial_search_v2 = original_initial_search
        Dataset.MIN_ROWS_COUNT = old_min_rows_count


def test_eval_sets_imbalanced_binary_classification(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    # Create imbalanced binary classification data (90% class 0, 10% class 1)
    np.random.seed(42)
    train_size = 1000
    eval_size = 200

    train_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size)],
            "feature1": np.random.randn(train_size),
        }
    )
    train_y = pd.Series(np.random.choice([0, 1], size=train_size, p=[0.9, 0.1]))

    eval_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size, train_size + eval_size)],
            "feature1": np.random.randn(eval_size),
        }
    )
    eval_y = pd.Series(np.random.choice([0, 1], size=eval_size, p=[0.9, 0.1]))

    result_wrapper = DataFrameWrapper()

    def mocked_initial_search(self, trace_id, file_path, metadata, metrics, search_customization):
        result_wrapper.df = pd.read_parquet(file_path)
        raise TestException

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    old_min_rows_count = Dataset.MIN_ROWS_COUNT
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher = FeaturesEnricher(
            search_keys={"phone": SearchKey.PHONE},
            endpoint=url,
            api_key="fake_api_key",
            logs_enabled=False,
        )

        try:
            enricher.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], calculate_metrics=False)
        except TestException:
            pass

        # Verify eval set is included in uploaded dataframe
        uploaded_df = result_wrapper.df
        assert uploaded_df is not None
        assert EVAL_SET_INDEX in uploaded_df.columns

        # Check train set (index 0) - should be balanced
        train_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 0]
        assert len(train_rows) > 0, "Train set should be included"
        # Train set should be balanced (undersampled)
        train_class_counts = train_rows[TARGET].value_counts()
        assert len(train_class_counts) == 2, "Train set should have both classes after balancing"

        # Check eval set (index 1) - should NOT be balanced
        eval_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 1]
        assert len(eval_rows) > 0, "Eval set should be included"
        assert not eval_rows[TARGET].isna().all(), "Eval set should have targets"
        # Eval set should maintain original imbalance
        eval_class_counts = eval_rows[TARGET].value_counts()
        assert len(eval_class_counts) == 2, "Eval set should have both classes"

    finally:
        _RestClient.initial_search_v2 = original_initial_search
        Dataset.MIN_ROWS_COUNT = old_min_rows_count


def test_eval_sets_regression(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    # Create regression data
    np.random.seed(42)
    train_size = 1000
    eval_size = 200

    train_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size)],
            "feature1": np.random.randn(train_size),
        }
    )
    train_y = pd.Series(np.random.randn(train_size))

    eval_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size, train_size + eval_size)],
            "feature1": np.random.randn(eval_size),
        }
    )
    eval_y = pd.Series(np.random.randn(eval_size))

    result_wrapper = DataFrameWrapper()

    def mocked_initial_search(self, trace_id, file_path, metadata, metrics, search_customization):
        result_wrapper.df = pd.read_parquet(file_path)
        raise TestException

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    old_min_rows_count = Dataset.MIN_ROWS_COUNT
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher = FeaturesEnricher(
            search_keys={"phone": SearchKey.PHONE},
            endpoint=url,
            api_key="fake_api_key",
            logs_enabled=False,
        )

        try:
            enricher.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], calculate_metrics=False)
        except TestException:
            pass

        # Verify eval set is included in uploaded dataframe
        uploaded_df = result_wrapper.df
        assert uploaded_df is not None
        assert EVAL_SET_INDEX in uploaded_df.columns

        # Check train set (index 0)
        train_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 0]
        assert len(train_rows) > 0, "Train set should be included"

        # Check eval set (index 1)
        eval_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 1]
        assert len(eval_rows) > 0, "Eval set should be included"
        assert not eval_rows[TARGET].isna().all(), "Eval set should have targets"

    finally:
        _RestClient.initial_search_v2 = original_initial_search
        Dataset.MIN_ROWS_COUNT = old_min_rows_count


def test_eval_sets_time_series_regression(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    # Create time series regression data
    np.random.seed(42)
    train_size = 500
    eval_size = 100

    dates_train = pd.date_range(start="2020-01-01", periods=train_size, freq="D")
    dates_eval = pd.date_range(start=dates_train[-1] + pd.Timedelta(days=1), periods=eval_size, freq="D")

    train_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size)],
            "date": dates_train,
            "feature1": np.random.randn(train_size),
        }
    )
    train_y = pd.Series(np.random.randn(train_size))

    eval_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size, train_size + eval_size)],
            "date": dates_eval,
            "feature1": np.random.randn(eval_size),
        }
    )
    eval_y = pd.Series(np.random.randn(eval_size))

    result_wrapper = DataFrameWrapper()

    def mocked_initial_search(self, trace_id, file_path, metadata, metrics, search_customization):
        result_wrapper.df = pd.read_parquet(file_path)
        raise TestException

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    old_min_rows_count = Dataset.MIN_ROWS_COUNT
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher = FeaturesEnricher(
            search_keys={"phone": SearchKey.PHONE, "date": SearchKey.DATE},
            endpoint=url,
            api_key="fake_api_key",
            logs_enabled=False,
        )

        try:
            enricher.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], calculate_metrics=False)
        except TestException:
            pass

        # Verify eval set is included in uploaded dataframe
        uploaded_df = result_wrapper.df
        assert uploaded_df is not None
        assert EVAL_SET_INDEX in uploaded_df.columns

        # Check train set (index 0)
        train_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 0]
        assert len(train_rows) > 0, "Train set should be included"

        # Check eval set (index 1)
        eval_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 1]
        assert len(eval_rows) > 0, "Eval set should be included"
        assert not eval_rows[TARGET].isna().all(), "Eval set should have targets"

        # For time series, verify that train and eval are sampled separately
        # (eval dates should be after train dates)
        if "date" in uploaded_df.columns:
            train_dates = train_rows["date"]
            eval_dates = eval_rows["date"]
            assert train_dates.max() < eval_dates.min(), "Eval dates should be after train dates"

    finally:
        _RestClient.initial_search_v2 = original_initial_search
        Dataset.MIN_ROWS_COUNT = old_min_rows_count


def test_eval_sets_with_oot(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    # Create train data
    np.random.seed(42)
    train_size = 1000
    eval_size = 200
    oot_size = 150

    train_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size)],
            "feature1": np.random.randn(train_size),
        }
    )
    train_y = pd.Series(np.random.choice([0, 1], size=train_size, p=[0.5, 0.5]))

    # Create eval set with targets
    eval_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size, train_size + eval_size)],
            "feature1": np.random.randn(eval_size),
        }
    )
    eval_y = pd.Series(np.random.choice([0, 1], size=eval_size, p=[0.5, 0.5]))

    # Create OOT eval set (without targets)
    oot_X = pd.DataFrame(
        {
            "phone": [f"+123456789{i:03d}" for i in range(train_size + eval_size, train_size + eval_size + oot_size)],
            "feature1": np.random.randn(oot_size),
        }
    )

    result_wrapper = DataFrameWrapper()

    def mocked_initial_search(self, trace_id, file_path, metadata, metrics, search_customization):
        result_wrapper.df = pd.read_parquet(file_path)
        raise TestException

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    old_min_rows_count = Dataset.MIN_ROWS_COUNT
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher = FeaturesEnricher(
            search_keys={"phone": SearchKey.PHONE},
            endpoint=url,
            api_key="fake_api_key",
            logs_enabled=False,
        )

        try:
            # Pass eval set with targets and OOT eval set without targets
            enricher.fit(
                train_X,
                train_y,
                eval_set=[(eval_X, eval_y), (oot_X,)],
                calculate_metrics=False,
            )
        except TestException:
            pass

        # Verify all sets are included in uploaded dataframe
        uploaded_df = result_wrapper.df
        assert uploaded_df is not None
        assert EVAL_SET_INDEX in uploaded_df.columns

        # Check train set (index 0)
        train_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 0]
        assert len(train_rows) > 0, "Train set should be included"
        assert len(train_rows) <= train_size, "Train set may be downsampled but should exist"
        assert not train_rows[TARGET].isna().any(), "Train set should have all targets"

        # Check eval set with targets (index 1)
        eval_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 1]
        assert len(eval_rows) > 0, "Eval set with targets should be included"
        assert len(eval_rows) <= eval_size, "Eval set may be downsampled but should exist"
        assert not eval_rows[TARGET].isna().any(), "Eval set with targets should have all targets"

        # Check OOT eval set (index 2) - should have NaN targets
        oot_rows = uploaded_df[uploaded_df[EVAL_SET_INDEX] == 2]
        assert len(oot_rows) > 0, "OOT eval set should be included"
        assert len(oot_rows) <= oot_size, "OOT eval set may be downsampled but should exist"
        assert oot_rows[TARGET].isna().all(), "OOT eval set targets should all be NaN"

        # Verify that train, eval, and OOT are sampled separately
        # (they should maintain their separate identities)
        unique_eval_indices = uploaded_df[EVAL_SET_INDEX].unique()
        assert 0 in unique_eval_indices, "Train set (index 0) should be present"
        assert 1 in unique_eval_indices, "Eval set (index 1) should be present"
        assert 2 in unique_eval_indices, "OOT eval set (index 2) should be present"

    finally:
        _RestClient.initial_search_v2 = original_initial_search
        Dataset.MIN_ROWS_COUNT = old_min_rows_count
