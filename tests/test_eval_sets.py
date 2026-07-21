import numpy as np
import pandas as pd
from requests_mock.mocker import Mocker

from upgini.dataset import Dataset
from upgini.features_enricher import FeaturesEnricher
from upgini.http import _RestClient
from upgini.metadata import EVAL_SET_INDEX, TARGET, SearchKey, ENTITY_SYSTEM_RECORD_ID, AddInfo
from unittest.mock import MagicMock

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


def test_find_fit_dataset_index_for_train_and_oot(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    train_y = pd.Series([0, 1])
    oot_X = pd.DataFrame({"phone": ["+10000000003", "+10000000004"], "f": [3.0, 4.0]})
    eval_X = pd.DataFrame({"phone": ["+10000000005", "+10000000006"], "f": [5.0, 6.0]})
    eval_y = pd.Series([1, 0])

    enricher.X = train_X
    enricher.y = train_y
    enricher.eval_set = enricher._check_eval_set([(eval_X, eval_y), oot_X], train_X)

    assert enricher._find_fit_dataset_index(train_X) == 0
    assert enricher._find_fit_dataset_index(eval_X) == 1
    assert enricher._find_fit_dataset_index(oot_X) == 2
    assert enricher._find_fit_dataset_index(train_X.copy()) is None
    assert enricher._find_fit_dataset_index(oot_X.copy()) is None


def test_transform_from_fit_reuses_enrichment_for_train_and_oot(requests_mock: Mocker):
    """transform(train) / transform(oot) after fit should reuse fit features, not validation search."""
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    oot_X = pd.DataFrame({"phone": ["+10000000003", "+10000000004"], "f": [3.0, 4.0]}, index=[10, 11])

    enricher.X = train_X
    enricher.y = pd.Series([0, 1])
    enricher.eval_set = enricher._check_eval_set([oot_X], train_X)
    enricher.feature_names_ = ["ads_feature"]
    enricher.external_source_feature_names = ["ads_feature"]
    enricher.fit_columns_renaming = {"phone_abc": "phone", "f_def": "f"}
    enricher.fit_search_keys = {"phone_abc": SearchKey.PHONE}
    enricher.fit_generated_features = []
    enricher.fit_select_features = False
    enricher.country_added = False
    enricher.add_info = AddInfo()

    # Simulate df kept on fit: hashed column names + entity ids + eval_set_index
    df_fit = pd.DataFrame(
        {
            "phone_abc": ["+10000000001", "+10000000002", "+10000000003", "+10000000004"],
            "f_def": [1.0, 2.0, 3.0, 4.0],
            TARGET: [0.0, 1.0, np.nan, np.nan],
            EVAL_SET_INDEX: [0, 0, 1, 1],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0, 201.0, 202.0],
        },
        index=[0, 1, 10, 11],
    )
    enricher.df_with_original_index = df_fit

    fit_features = pd.DataFrame(
        {
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0, 201.0, 202.0],
            "ads_feature": [10.0, 20.0, 30.0, 40.0],
        }
    )

    search_task = MagicMock()
    fm = MagicMock()
    fm.name = "ads_feature"
    fm.shap_value = 1.0
    fm.source = "ads"
    fm.from_online_api = False
    search_task.get_all_features_metadata_v2.return_value = [fm]
    search_task.get_all_initial_raw_features.return_value = fit_features
    col_phone = MagicMock(originalName="phone", name="phone_abc")
    col_f = MagicMock(originalName="f", name="f_def")
    search_task.get_file_metadata.return_value = MagicMock(columns=[col_phone, col_f], droppedColumns=[])
    enricher._search_task = search_task
    enricher.search_id = "fake_search"

    def fail_if_validation(*args, **kwargs):
        raise AssertionError("validation search should not be called when reusing fit enrichment")

    original_validation = Dataset.validation
    Dataset.validation = fail_if_validation
    try:
        enriched_train = enricher.transform(train_X, keep_input=True)
        assert enriched_train is not None
        assert len(enriched_train) == len(train_X)
        assert "ads_feature" in enriched_train.columns
        assert list(enriched_train["ads_feature"]) == [10.0, 20.0]

        enriched_oot = enricher.transform(oot_X, keep_input=True)
        assert enriched_oot is not None
        assert len(enriched_oot) == len(oot_X)
        assert "ads_feature" in enriched_oot.columns
        assert list(enriched_oot["ads_feature"]) == [30.0, 40.0]
    finally:
        Dataset.validation = original_validation


def test_metrics_cache_key_isolates_outliers_and_exclude(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    from upgini.metadata import ModelTaskType

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        model_task_type=ModelTaskType.REGRESSION,
    )

    X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    y = pd.Series([0.1, 0.2])

    key_default = enricher._get_metrics_cache_key(X, y, None)
    key_keep_outliers = enricher._get_metrics_cache_key(X, y, None, remove_outliers_calc_metrics=False)
    key_exclude = enricher._get_metrics_cache_key(X, y, None, exclude_features_sources=["ads_a"])
    key_exclude_other_order = enricher._get_metrics_cache_key(
        X, y, None, exclude_features_sources=["ads_b", "ads_a"]
    )
    key_exclude_sorted = enricher._get_metrics_cache_key(
        X, y, None, exclude_features_sources=["ads_a", "ads_b"]
    )

    assert key_default != key_keep_outliers
    assert key_default != key_exclude
    assert key_keep_outliers != key_exclude
    assert key_exclude_other_order == key_exclude_sorted
    assert "outliers_removed=True" in key_default
    assert "outliers_removed=False" in key_keep_outliers
    assert "exclude=ads_a" in key_exclude
    assert "exclude_oot=False" in key_default

    key_exclude_oot = enricher._get_metrics_cache_key(X, y, None, exclude_oot=True)
    assert key_default != key_exclude_oot
    assert "exclude_oot=True" in key_exclude_oot

    # Non-regression: explicit outlier flags are no-ops and share a key
    enricher.model_task_type = ModelTaskType.BINARY
    binary_none = enricher._get_metrics_cache_key(X, y, None, remove_outliers_calc_metrics=None)
    binary_true = enricher._get_metrics_cache_key(X, y, None, remove_outliers_calc_metrics=True)
    binary_false = enricher._get_metrics_cache_key(X, y, None, remove_outliers_calc_metrics=False)
    assert binary_none == binary_true == binary_false
    assert "outliers_removed=False" in binary_none


def test_metrics_cache_exclude_does_not_poison_full_entry(requests_mock: Mocker):
    """Reading with exclude from a full cache must not overwrite the full entry."""
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    from upgini.metadata import ModelTaskType

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        model_task_type=ModelTaskType.BINARY,
    )

    X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    y = pd.Series([0, 1])
    enricher.X = X
    enricher.y = y
    enricher.eval_set = []

    full_hash = enricher._get_metrics_cache_key(X, y, None)
    exclude_hash = enricher._get_metrics_cache_key(X, y, None, exclude_features_sources=["ads_feature"])

    X_sampled = X.copy()
    y_sampled = y.copy()
    enriched_full = X.copy()
    enriched_full["ads_feature"] = [10.0, 20.0]
    cache = enricher._FeaturesEnricher__cached_sampled_datasets
    cache[full_hash] = (
        X_sampled,
        y_sampled,
        enriched_full,
        {},
        {},
        {},
        [],
    )

    result = enricher._FeaturesEnricher__get_sampled_cached_enriched(
        full_hash,
        exclude_features_sources=["ads_feature"],
        write_hash=exclude_hash,
    )
    assert "ads_feature" not in result.enriched_X.columns
    # Full cache entry must remain intact
    assert "ads_feature" in cache[full_hash][2].columns
    # Excluded view may be stored under its own key
    assert exclude_hash in cache
    assert "ads_feature" not in cache[exclude_hash][2].columns


def test_metrics_cache_derives_outliers_removed_from_psi_entry(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    from upgini.metadata import ModelTaskType

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        model_task_type=ModelTaskType.REGRESSION,
    )

    X = pd.DataFrame(
        {
            "phone": ["+10000000001", "+10000000002", "+10000000003"],
            "f": [1.0, 2.0, 3.0],
            ENTITY_SYSTEM_RECORD_ID: [101, 102, 103],
        }
    )
    y = pd.Series([0.1, 0.2, 99.0], name=TARGET)
    enricher.X = X.drop(columns=[ENTITY_SYSTEM_RECORD_ID])
    enricher.y = y
    enricher.eval_set = []
    enricher.feature_names_ = ["ads_feature"]
    enricher.df_with_original_index = X.assign(**{TARGET: y.values})

    search_task = MagicMock()
    search_task.get_target_outliers.return_value = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [103]})
    enricher._search_task = search_task

    keep_hash = enricher._get_metrics_cache_key(enricher.X, enricher.y, None, remove_outliers_calc_metrics=False)
    drop_hash = enricher._get_metrics_cache_key(enricher.X, enricher.y, None, remove_outliers_calc_metrics=None)
    assert keep_hash != drop_hash

    X_sampled = X.copy()
    y_sampled = y.copy()
    enriched = X.copy()
    enriched["ads_feature"] = [10.0, 20.0, 30.0]
    cache = enricher._FeaturesEnricher__cached_sampled_datasets
    cache[keep_hash] = (
        X_sampled,
        y_sampled,
        enriched,
        {},
        {"phone": SearchKey.PHONE},
        {},
        [],
    )

    result = enricher._get_enriched_datasets(
        validated_X=enricher.X,
        validated_y=enricher.y,
        validated_eval_set=None,
        exclude_features_sources=None,
        is_input_same_as_fit=True,
        is_demo_dataset=False,
        remove_outliers_calc_metrics=None,
        progress_bar=None,
        progress_callback=None,
    )

    assert len(result.enriched_X) == 2
    assert 103 not in set(result.enriched_X[ENTITY_SYSTEM_RECORD_ID])
    assert len(result.X_sampled) == 2
    assert len(result.y_sampled) == 2
    # Keep-outliers (PSI) entry must remain intact
    assert len(cache[keep_hash][2]) == 3
    assert 103 in set(cache[keep_hash][2][ENTITY_SYSTEM_RECORD_ID])
    # Outliers-removed view stored under its own key
    assert drop_hash in cache
    assert len(cache[drop_hash][2]) == 2
    assert 103 not in set(cache[drop_hash][2][ENTITY_SYSTEM_RECORD_ID])


def test_metrics_cache_uses_validated_key_not_raw(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    from upgini.features_enricher import hash_input
    from upgini.metadata import ModelTaskType

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        model_task_type=ModelTaskType.BINARY,
        exclude_columns=["noise"],
    )

    raw_X = pd.DataFrame(
        {
            "phone": ["+10000000001", "+10000000002"],
            "f": [1.0, 2.0],
            "noise": [9.0, 9.0],
        }
    )
    y = pd.Series([0, 1])
    validated_X = enricher._validate_X(raw_X)
    assert "noise" not in validated_X.columns
    assert hash_input(raw_X, y) != hash_input(validated_X, y)

    raw_key = enricher._get_metrics_cache_key(raw_X, y, None)
    validated_key = enricher._get_metrics_cache_key(validated_X, y, None)
    assert raw_key != validated_key

    enricher.X = raw_X
    enricher.y = y
    enricher.eval_set = []
    enricher.feature_names_ = ["ads_feature"]
    enricher.df_with_original_index = validated_X.assign(**{TARGET: y.values, ENTITY_SYSTEM_RECORD_ID: [1, 2]})

    from_fit_calls = []

    def fake_from_fit(vX, vY, eval_set, remove_outliers_calc_metrics, datasets_hash):
        from_fit_calls.append(datasets_hash)
        X_sampled = vX.copy()
        y_sampled = vY.copy()
        enriched = vX.copy()
        enriched["ads_feature"] = [10.0, 20.0]
        return enricher._FeaturesEnricher__cache_and_return_results(
            datasets_hash,
            X_sampled,
            y_sampled,
            enriched,
            {},
            {},
            {"phone": SearchKey.PHONE},
            [],
        )

    enricher._FeaturesEnricher__get_enriched_from_fit = fake_from_fit

    kwargs = dict(
        validated_X=validated_X,
        validated_y=y,
        validated_eval_set=None,
        exclude_features_sources=None,
        is_input_same_as_fit=True,
        is_demo_dataset=False,
        remove_outliers_calc_metrics=None,
        progress_bar=None,
        progress_callback=None,
    )
    result1 = enricher._get_enriched_datasets(**kwargs)
    result2 = enricher._get_enriched_datasets(**kwargs)

    cache = enricher._FeaturesEnricher__cached_sampled_datasets
    assert from_fit_calls == [validated_key]
    assert validated_key in cache
    assert raw_key not in cache
    assert "ads_feature" in result1.enriched_X.columns
    assert "ads_feature" in result2.enriched_X.columns


def test_metrics_cache_as_input_uses_validated_key(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    from upgini.metadata import ModelTaskType

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        model_task_type=ModelTaskType.BINARY,
    )

    X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    y = pd.Series([0, 1])
    eval_X = pd.DataFrame({"phone": ["+10000000003"], "f": [3.0]})
    eval_y = pd.Series([0])
    eval_set = [(eval_X, eval_y)]

    enricher.X = X
    enricher.y = y
    enricher.eval_set = eval_set
    enricher.feature_names_ = []
    enricher.add_info = AddInfo()

    expected_key = enricher._get_metrics_cache_key(X, y, eval_set)
    assert expected_key != ""

    as_input_calls = []
    original_as_input = enricher._FeaturesEnricher__get_enriched_as_input

    def counting_as_input(validated_X, validated_y, validated_eval_set, is_demo_dataset, datasets_hash):
        as_input_calls.append(datasets_hash)
        return original_as_input(validated_X, validated_y, validated_eval_set, is_demo_dataset, datasets_hash)

    enricher._FeaturesEnricher__get_enriched_as_input = counting_as_input

    kwargs = dict(
        validated_X=X,
        validated_y=y,
        validated_eval_set=eval_set,
        exclude_features_sources=None,
        is_input_same_as_fit=True,
        is_demo_dataset=False,
        remove_outliers_calc_metrics=None,
        progress_bar=None,
        progress_callback=None,
    )
    result1 = enricher._get_enriched_datasets(**kwargs)
    result2 = enricher._get_enriched_datasets(**kwargs)

    cache = enricher._FeaturesEnricher__cached_sampled_datasets
    assert as_input_calls == [expected_key]
    assert expected_key in cache
    assert "" not in cache
    assert len(result1.eval_set_sampled_dict) == 1
    assert len(result2.eval_set_sampled_dict) == 1


def test_metrics_cache_exclude_oot_does_not_poison_full_entry(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    from upgini.metadata import ModelTaskType

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        model_task_type=ModelTaskType.BINARY,
    )

    X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    y = pd.Series([0, 1])
    eval_X = pd.DataFrame({"phone": ["+10000000003"], "f": [3.0]})
    eval_y = pd.Series([0])
    oot_X = pd.DataFrame({"phone": ["+10000000004"], "f": [4.0]})
    oot_y = pd.Series([np.nan])
    eval_set = [(eval_X, eval_y), (oot_X, oot_y)]

    enricher.X = X
    enricher.y = y
    enricher.eval_set = eval_set
    enricher.feature_names_ = ["ads_feature"]

    with_oot_hash = enricher._get_metrics_cache_key(X, y, eval_set, exclude_oot=False)
    without_oot_hash = enricher._get_metrics_cache_key(X, y, eval_set, exclude_oot=True)
    assert with_oot_hash != without_oot_hash

    X_sampled = X.copy()
    y_sampled = y.copy()
    enriched = X.copy()
    enriched["ads_feature"] = [10.0, 20.0]
    eval_enriched = eval_X.copy()
    eval_enriched["ads_feature"] = [30.0]
    oot_enriched = oot_X.copy()
    oot_enriched["ads_feature"] = [40.0]
    cache = enricher._FeaturesEnricher__cached_sampled_datasets
    cache[with_oot_hash] = (
        X_sampled,
        y_sampled,
        enriched,
        {
            0: (eval_X.copy(), eval_enriched, eval_y.copy()),
            1: (oot_X.copy(), oot_enriched, oot_y.copy()),
        },
        {"phone": SearchKey.PHONE},
        {},
        [],
    )

    result = enricher._get_enriched_datasets(
        validated_X=X,
        validated_y=y,
        validated_eval_set=eval_set,
        exclude_features_sources=None,
        is_input_same_as_fit=True,
        is_demo_dataset=False,
        remove_outliers_calc_metrics=None,
        progress_bar=None,
        progress_callback=None,
        exclude_oot=True,
    )

    assert 1 not in result.eval_set_sampled_dict
    assert 0 in result.eval_set_sampled_dict
    assert len(result.eval_set_sampled_dict[0][0]) == 1
    # With-OOT (PSI) entry must remain intact
    assert 1 in cache[with_oot_hash][3]
    assert len(cache[with_oot_hash][3][1][0]) == 1
    # Metrics view stored under exclude_oot=True without empty OOT slot
    assert without_oot_hash in cache
    assert 1 not in cache[without_oot_hash][3]


def test_metrics_cache_derives_multi_dimension_from_fullest_entry(requests_mock: Mocker):
    """PSI fullest entry must satisfy metrics that need outliers removed and OOT dropped together."""
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    from upgini.metadata import ModelTaskType

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        model_task_type=ModelTaskType.REGRESSION,
    )

    X = pd.DataFrame(
        {
            "phone": ["+10000000001", "+10000000002", "+10000000003"],
            "f": [1.0, 2.0, 3.0],
            ENTITY_SYSTEM_RECORD_ID: [101, 102, 103],
        }
    )
    y = pd.Series([0.1, 0.2, 99.0], name=TARGET)
    eval_X = pd.DataFrame({"phone": ["+10000000004"], "f": [4.0], ENTITY_SYSTEM_RECORD_ID: [104]})
    eval_y = pd.Series([0.5])
    oot_X = pd.DataFrame({"phone": ["+10000000005"], "f": [5.0], ENTITY_SYSTEM_RECORD_ID: [105]})
    oot_y = pd.Series([np.nan])
    eval_set = [(eval_X.drop(columns=[ENTITY_SYSTEM_RECORD_ID]), eval_y), (oot_X.drop(columns=[ENTITY_SYSTEM_RECORD_ID]), oot_y)]

    enricher.X = X.drop(columns=[ENTITY_SYSTEM_RECORD_ID])
    enricher.y = y
    enricher.eval_set = eval_set
    enricher.feature_names_ = ["ads_feature"]
    enricher.df_with_original_index = X.assign(**{TARGET: y.values})

    search_task = MagicMock()
    search_task.get_target_outliers.return_value = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [103]})
    enricher._search_task = search_task

    fullest_hash = enricher._get_metrics_cache_key(
        enricher.X, y, eval_set, remove_outliers_calc_metrics=False, exclude_oot=False
    )
    wanted_hash = enricher._get_metrics_cache_key(
        enricher.X, y, eval_set, remove_outliers_calc_metrics=None, exclude_oot=True
    )
    assert fullest_hash != wanted_hash
    assert "outliers_removed=False" in fullest_hash
    assert "exclude_oot=False" in fullest_hash
    assert "outliers_removed=True" in wanted_hash
    assert "exclude_oot=True" in wanted_hash

    enriched = X.copy()
    enriched["ads_feature"] = [10.0, 20.0, 30.0]
    eval_enriched = eval_X.copy()
    eval_enriched["ads_feature"] = [40.0]
    oot_enriched = oot_X.copy()
    oot_enriched["ads_feature"] = [50.0]
    cache = enricher._FeaturesEnricher__cached_sampled_datasets
    cache[fullest_hash] = (
        X.copy(),
        y.copy(),
        enriched,
        {
            0: (eval_X.copy(), eval_enriched, eval_y.copy()),
            1: (oot_X.copy(), oot_enriched, oot_y.copy()),
        },
        {"phone": SearchKey.PHONE},
        {},
        [],
    )

    result = enricher._get_enriched_datasets(
        validated_X=enricher.X,
        validated_y=y,
        validated_eval_set=eval_set,
        exclude_features_sources=None,
        is_input_same_as_fit=True,
        is_demo_dataset=False,
        remove_outliers_calc_metrics=None,
        progress_bar=None,
        progress_callback=None,
        exclude_oot=True,
    )

    assert len(result.enriched_X) == 2
    assert 103 not in set(result.enriched_X[ENTITY_SYSTEM_RECORD_ID])
    assert 1 not in result.eval_set_sampled_dict
    assert 0 in result.eval_set_sampled_dict
    # Fullest PSI entry unchanged
    assert len(cache[fullest_hash][2]) == 3
    assert 1 in cache[fullest_hash][3]
    assert wanted_hash in cache
    assert len(cache[wanted_hash][2]) == 2
    assert 1 not in cache[wanted_hash][3]


def test_extract_eval_data_skips_empty_oot_when_requested(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )
    enriched_df = pd.DataFrame(
        {
            "phone": ["+1", "+2", "+3"],
            "f": [1.0, 2.0, 3.0],
            TARGET: [0.0, 1.0, np.nan],
            EVAL_SET_INDEX: [0, 1, 2],
        }
    )
    # Eval index 2 is OOT (all-NaN target) and would be empty if stripped from df
    enriched_without_oot = enriched_df[enriched_df[EVAL_SET_INDEX] != 2]
    x_columns = ["phone", "f"]
    result = enricher._FeaturesEnricher__extract_eval_data(
        enriched_without_oot,
        x_columns,
        x_columns,
        eval_set_len=2,
        skip_empty_or_oot=True,
    )
    assert list(result.keys()) == [0]
    assert len(result[0][0]) == 1
