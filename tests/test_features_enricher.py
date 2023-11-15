import os
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from requests_mock.mocker import Mocker

from upgini import FeaturesEnricher, SearchKey
from upgini.dataset import Dataset
from upgini.errors import ValidationError
from upgini.http import _RestClient
from upgini.metadata import (
    CVType,
    FeaturesMetadataV2,
    HitRateMetrics,
    ModelEvalSet,
    ModelTaskType,
    ProviderTaskMetadataV2,
    RuntimeParameters,
)
from upgini.resource_bundle import bundle
from upgini.search_task import SearchTask
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter

from .utils import (
    mock_default_requests,
    mock_get_metadata,
    mock_get_task_metadata_v2,
    mock_initial_and_validation_summary,
    mock_initial_progress,
    mock_initial_search,
    mock_initial_summary,
    mock_raw_features,
    mock_validation_progress,
    mock_validation_raw_features,
    mock_validation_search,
    mock_validation_summary,
)

segment_header = bundle.get("quality_metrics_segment_header")
train_segment = bundle.get("quality_metrics_train_segment")
eval_1_segment = bundle.get("quality_metrics_eval_segment").format(1)
eval_2_segment = bundle.get("quality_metrics_eval_segment").format(2)
match_rate_header = bundle.get("quality_metrics_match_rate_header")
rows_header = bundle.get("quality_metrics_rows_header")
target_mean_header = bundle.get("quality_metrics_mean_target_header")
baseline_rocauc = bundle.get("quality_metrics_baseline_header").format("roc_auc")
enriched_rocauc = bundle.get("quality_metrics_enriched_header").format("roc_auc")
baseline_gini = bundle.get("quality_metrics_baseline_header").format("GINI")
enriched_gini = bundle.get("quality_metrics_enriched_header").format("GINI")
uplift = bundle.get("quality_metrics_uplift_header")
feature_name_header = bundle.get("features_info_name")
shap_value_header = bundle.get("features_info_shap")
hitrate_header = bundle.get("features_info_hitrate")

SearchTask.PROTECT_FROM_RATE_LIMIT = False
SearchTask.POLLING_DELAY_SECONDS = 0.1
pd.set_option("mode.chained_assignment", "raise")
pd.set_option("display.max_columns", 1000)


def test_search_keys_validation(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    error_message = bundle.get("date_and_datetime_simultanious")
    with pytest.raises(Exception, match=error_message):
        FeaturesEnricher(
            search_keys={"d1": SearchKey.DATE, "dt2": SearchKey.DATETIME},
            endpoint=url,
            logs_enabled=False,
        )

    error_message = bundle.get("postal_code_without_country")
    with pytest.raises(Exception, match=error_message):
        FeaturesEnricher(search_keys={"postal_code": SearchKey.POSTAL_CODE}, endpoint=url, logs_enabled=False)


def test_features_enricher(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_initial_progress(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    ads_search_task_id = mock_initial_and_validation_summary(
        requests_mock,
        url,
        search_task_id,
        validation_search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].reset_index(drop=True)
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
    )
    assert enriched_train_features.shape == (10000, 3)

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
    )
    assert enriched_train_features.shape == (10000, 3)

    metrics = enricher.calculate_metrics()

    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment, eval_1_segment, eval_2_segment],
            rows_header: [10000, 1000, 1000],
            target_mean_header: [0.5044, 0.487, 0.486],
            enriched_gini: [0.006920, 0.007322, 0.007022],
        }
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 1
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1


def test_eval_set_with_diff_order_of_columns(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(requests_mock, url, search_task_id)
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    # shuffle columns
    eval1_features = eval1_features[set(eval1_features.columns)]
    eval1_target = eval1_df["target"].reset_index(drop=True)

    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    # Add feature that doesn't exist in train df
    eval2_features["new_feature"] = "test"
    eval2_target = eval2_df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    with pytest.raises(ValidationError, match=bundle.get("eval_x_and_x_diff_shape")):
        enricher.fit(
            train_features,
            train_target,
            eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
            calculate_metrics=False,
            keep_input=True,
        )

    enricher.fit(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target)],
        calculate_metrics=False,
        keep_input=True,
    )


def test_features_enricher_with_index_and_column_same_names(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_initial_progress(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    ads_search_task_id = mock_initial_and_validation_summary(
        requests_mock,
        url,
        search_task_id,
        validation_search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    df = df.drop_duplicates(subset="rep_date")
    df = df.set_index("rep_date")
    df["rep_date"] = df.index
    train_features = df.drop(columns="target")
    train_target = df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    min_rows_count = Dataset.MIN_ROWS_COUNT
    Dataset.MIN_ROWS_COUNT = 5
    try:
        enriched_train_features = enricher.fit_transform(
            train_features,
            train_target,
            calculate_metrics=False,
            keep_input=True,
        )
        assert enriched_train_features.shape == (6, 3)

        enriched_train_features = enricher.fit_transform(
            train_features,
            train_target,
            calculate_metrics=False,
            keep_input=False,
        )
        assert enriched_train_features.shape == (6, 1)
    finally:
        Dataset.MIN_ROWS_COUNT = min_rows_count


def test_saved_features_enricher(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/validation_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_initial_progress(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    ads_search_task_id = mock_initial_and_validation_summary(
        requests_mock,
        url,
        search_task_id,
        validation_search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="numeric", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features, metrics_calculation=True)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)
    mock_validation_raw_features(
        requests_mock, url, validation_search_task_id, path_to_mock_features, metrics_calculation=True
    )

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].reset_index(drop=True)
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        search_id=search_task_id,
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    enriched_train_features = enricher.transform(
        train_features,
    )
    print(enriched_train_features)
    assert enriched_train_features.shape == (10000, 3)

    metrics = enricher.calculate_metrics(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
    )
    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment, eval_1_segment, eval_2_segment],
            rows_header: [10000, 1000, 1000],
            target_mean_header: [0.5044, 0.487, 0.486],
            enriched_gini: [0.000779, 0.000780, -0.003822],
        }
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 1
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1


def test_features_enricher_with_demo_key(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"].to_frame()
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].to_frame().reset_index(drop=True)
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"].to_frame()

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        keep_input=True,
        calculate_metrics=False,
    )
    assert enriched_train_features.shape == (10000, 3)

    metrics = enricher.calculate_metrics()

    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment, eval_1_segment, eval_2_segment],
            rows_header: [10000, 1000, 1000],
            target_mean_header: [0.5044, 0.487, 0.486],
            baseline_gini: [0.009285, 0.066373, 0.049547],
            enriched_gini: [0.006104, 0.056032, -0.006200],
            uplift: [-0.003181, -0.010341, -0.055747],
        }
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 1
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1


def test_features_enricher_with_diff_size_xy(requests_mock: Mocker):
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"].to_frame()
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].to_frame()

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    with pytest.raises(ValidationError, match=bundle.get("x_and_y_diff_size").format(1000, 500)):
        enricher.fit(train_features.head(1000), train_target.head(500))

    with pytest.raises(ValidationError, match=bundle.get("x_and_y_diff_size_eval_set").format(1000, 500)):
        enricher.fit(train_features, train_target, [(eval1_features, eval1_target.head(500))])


def test_features_enricher_with_numpy(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000).reset_index(drop=True)
    train_features = train_df.drop(columns="target").values
    train_target = train_df["target"].values
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target").values
    eval1_target = eval1_df["target"].values
    eval2_df = df[11000:12000].reset_index(drop=True)
    eval2_features = eval2_df.drop(columns="target").values
    eval2_target = eval2_df["target"].values

    enricher = FeaturesEnricher(
        search_keys={0: SearchKey.PHONE, 1: SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
    )
    assert enriched_train_features.shape == (10000, 3)

    metrics = enricher.calculate_metrics()
    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment, eval_1_segment, eval_2_segment],
            rows_header: [10000, 1000, 1000],
            target_mean_header: [0.5044, 0.487, 0.486],
            enriched_gini: [0.006920, 0.007322, 0.007022],
        }
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 1
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1

    enricher.transform(train_features)


def test_features_enricher_with_named_index(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    df.index.name = "custom_index_name"
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"].to_list()
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].reset_index(drop=True).to_list()
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"].to_list()

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
    )
    assert enriched_train_features.shape == (10000, 3)
    assert enriched_train_features.index.name == "custom_index_name"

    metrics = enricher.calculate_metrics()
    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment, eval_1_segment, eval_2_segment],
            rows_header: [10000, 1000, 1000],
            target_mean_header: [0.5044, 0.487, 0.486],
            enriched_gini: [0.006920, 0.007322, 0.007022],
        }
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 1
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1


def test_features_enricher_with_index_column(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    df = df.reset_index()
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"].to_list()
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].reset_index(drop=True).to_list()
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"].to_list()

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
    )
    assert enriched_train_features.shape == (10000, 3)
    assert "index" not in enriched_train_features.columns

    metrics = enricher.calculate_metrics()
    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment, eval_1_segment, eval_2_segment],
            rows_header: [10000, 1000, 1000],
            target_mean_header: [0.5044, 0.487, 0.486],
            enriched_gini: [0.006920, 0.007322, 0.007022],
        }
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 1
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1


def test_features_enricher_with_complex_feature_names(requests_mock: Mocker):
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    requests_mock.get(
        url + f"/public/api/v2/search/{search_task_id}/metadata",
        json={
            "fileUploadId": "123",
            "fileMetadataId": "123",
            "name": "test",
            "description": "",
            "columns": [
                {
                    "index": 0,
                    "name": "phone_num",
                    "originalName": "phone_num",
                    "dataType": "STRING",
                    "meaningType": "MSISDN",
                },
                {
                    "index": 1,
                    "name": "cos_3_freq_w_sun_",
                    "originalName": "cos(3,freq=W-SUN)",
                    "dataType": "INT",
                    "meaningType": "FEATURE",
                },
                {"index": 2, "name": "target", "originalName": "target", "dataType": "INT", "meaningType": "TARGET"},
                {
                    "index": 3,
                    "name": "system_record_id",
                    "originalName": "system_record_id",
                    "dataType": "INT",
                    "meaningType": "SYSTEM_RECORD_ID",
                },
            ],
            "searchKeys": [["phone_num"]],
            "hierarchicalGroupKeys": [],
            "hierarchicalSubgroupKeys": [],
            "rowsCount": 5319,
        },
    )

    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="f_feature123", type="numerical", source="ads", hit_rate=99.0, shap_value=0.9),
                FeaturesMetadataV2(
                    name="cos_3_freq_w_sun_", type="numerical", source="etalon", hit_rate=100.0, shap_value=0.1
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=5319, max_hit_count=5266, hit_rate=0.99, hit_rate_percent=99.0
            ),
            features_used_for_embeddings=["cos_3_freq_w_sun_"],
        ),
    )
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/complex_feature_name_features.parquet"
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features, metrics_calculation=True)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/complex_feature_name_tds.parquet")
    df = pd.read_parquet(path)
    train_features = df.drop(columns="target")
    train_target = df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    enricher.fit(
        train_features,
        train_target,
        calculate_metrics=False,
    )

    metrics = enricher.calculate_metrics()
    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment],
            rows_header: [5319],
            target_mean_header: [0.6364],
            baseline_gini: [0.001041],
            enriched_gini: [0.006466],
            uplift: [0.005425],
        }
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    features_info = enricher.get_features_info()
    print(features_info)

    assert enricher.feature_names_ == ["f_feature123"]
    assert enricher.feature_importances_ == [0.9]
    assert len(features_info) == 1
    first_feature_info = features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "f_feature123"
    assert first_feature_info[shap_value_header] == 0.9
    assert first_feature_info[hitrate_header] == 99.0
    # Client features are no longer shown
    # second_feature_info = features_info.iloc[1]
    # assert second_feature_info[feature_name_header] == "cos(3,freq=W-SUN)"
    # assert second_feature_info[shap_value_header] == 0.1
    # assert second_feature_info[hitrate_header] == 100.0

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    original_validation = Dataset.validation

    def wrapped_validation(
        self,
        trace_id: str,
        initial_search_task_id: str,
        start_time: int,
        return_scores: bool = True,
        extract_features: bool = False,
        runtime_parameters: Optional[RuntimeParameters] = None,
        exclude_features_sources: Optional[List[str]] = None,
        metrics_calculation: bool = False,
        silent_mode: bool = False,
        progress_bar=None,
        progress_callback=None,
    ):
        assert "cos(3,freq=W-SUN)" in self.data.columns
        assert runtime_parameters.properties["features_for_embeddings"] == "cos_3_freq_w_sun_"
        return original_validation(
            self,
            trace_id,
            initial_search_task_id,
            start_time,
            return_scores,
            extract_features,
            runtime_parameters,
            exclude_features_sources,
            metrics_calculation,
            silent_mode,
            progress_bar,
            progress_callback,
        )

    Dataset.validation = wrapped_validation

    try:
        transformed = enricher.transform(train_features)
        print(transformed)
    finally:
        Dataset.validation = original_validation


def test_features_enricher_fit_transform_runtime_parameters(requests_mock: Mocker):
    url = "http://fake_url2"
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(
                    name="feature",
                    type="NUMERIC",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                    commercial_schema="Trial",
                    data_provider="Upgini",
                    data_provider_link="https://upgini.com",
                    data_source="Community shared",
                    data_source_link="https://upgini.com",
                ),
                FeaturesMetadataV2(
                    name="SystemRecordId_473310000", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=1.0
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]
    eval1_df = df[10000:11000]
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"]
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        date_format="%Y-%m-%d",
        endpoint=url,
        api_key="fake_api_key",
        runtime_parameters=RuntimeParameters(properties={"runtimeProperty1": "runtimeValue1"}),
        logs_enabled=False,
    )
    assert enricher.runtime_parameters is not None

    enricher.fit(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
    )

    fit_req = None
    initial_search_url = url + "/public/api/v2/search/initial"
    for elem in requests_mock.request_history:
        if elem.url == initial_search_url:
            fit_req = elem

    # TODO: can be better with
    #  https://metareal.blog/en/post/2020/05/03/validating-multipart-form-data-with-requests-mock/
    # It"s do-able to parse req with cgi module and verify contents
    assert fit_req is not None
    assert "runtimeProperty1" in str(fit_req.body)
    assert "runtimeValue1" in str(fit_req.body)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    transformed = enricher.transform(train_features, keep_input=True)

    transform_req = None
    transform_url = url + "/public/api/v2/search/validation?initialSearchTaskId=" + search_task_id
    for elem in requests_mock.request_history:
        if elem.url == transform_url:
            transform_req = elem

    assert transform_req is not None
    assert "runtimeProperty1" in str(transform_req.body)
    assert "runtimeValue1" in str(transform_req.body)

    assert transformed.shape == (10000, 4)


def test_features_enricher_fit_custom_loss(requests_mock: Mocker):
    url = "http://fake_url2"
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(
                    name="feature",
                    type="NUMERIC",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                    commercial_schema="Trial",
                    data_provider="Upgini",
                    data_provider_link="https://upgini.com",
                    data_source="Community shared",
                    data_source_link="https://upgini.com",
                ),
                FeaturesMetadataV2(
                    name="SystemRecordId_473310000", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=1.0
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]
    eval1_df = df[10000:11000]
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"]
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        date_format="%Y-%m-%d",
        endpoint=url,
        api_key="fake_api_key",
        loss="poisson",
        model_task_type=ModelTaskType.REGRESSION,
        logs_enabled=False,
    )

    enricher.fit(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
    )

    fit_req = None
    initial_search_url = url + "/public/api/v2/search/initial"
    for elem in requests_mock.request_history:
        if elem.url == initial_search_url:
            fit_req = elem

    assert fit_req is not None
    assert "lightgbm_params_preselection.objective" in str(fit_req.body)
    assert "lightgbm_params_base.objective" in str(fit_req.body)
    assert "lightgbm_params_segment.objective" in str(fit_req.body)
    assert "poisson" in str(fit_req.body)


def test_search_with_only_personal_keys(requests_mock: Mocker):
    url = "https://some.fake.url"

    mock_default_requests(requests_mock, url)

    df = pd.DataFrame(
        {
            "phone": ["1234567890", "2345678901", "3456789012"],
            "email": ["test1@gmail.com", "test2@gmail.com", "test3@gmail.com"],
            "target": [0, 1, 0],
        }
    )

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE, "email": SearchKey.EMAIL},
        endpoint=url,
        logs_enabled=False,
        raise_validation_error=True,
    )
    with pytest.raises(ValidationError):
        enricher.fit_transform(df.drop(columns="target"), df.target)


def test_filter_by_importance(requests_mock: Mocker):
    url = "https://some.fake.url"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)

    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=0.7)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000)
    print(train_df.head(10))
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]
    eval1_df = df[10000:11000]
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"]
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        date_format="%Y-%m-%d",
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    eval_set = [(eval1_features, eval1_target), (eval2_features, eval2_target)]

    enricher.fit(train_features, train_target, eval_set=eval_set, calculate_metrics=False)

    metrics = enricher.calculate_metrics(importance_threshold=0.8)

    assert metrics is None

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=eval_set,
        calculate_metrics=False,
        keep_input=True,
        importance_threshold=0.8,
    )

    assert train_features.shape == (10000, 2)

    test_features = enricher.transform(eval1_features, keep_input=True, importance_threshold=0.8)

    assert test_features.shape == (1000, 2)


def test_filter_by_max_features(requests_mock: Mocker):
    url = "https://some.fake.url"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)

    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=0.7)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]
    eval1_df = df[10000:11000]
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"]
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        date_format="%Y-%m-%d",
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    eval_set = [(eval1_features, eval1_target), (eval2_features, eval2_target)]

    enricher.fit(train_features, train_target, eval_set=eval_set, calculate_metrics=False)

    metrics = enricher.calculate_metrics(max_features=0)

    assert metrics is None

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    train_features = enricher.fit_transform(
        train_features, train_target, eval_set=eval_set, calculate_metrics=False, keep_input=True, max_features=0
    )

    assert train_features.shape == (10000, 2)

    test_features = enricher.transform(eval1_features, keep_input=True, max_features=0)

    assert test_features.shape == (1000, 2)


def test_validation_metrics_calculation(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)

    tds = pd.DataFrame({"date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)], "target": [0, 1, 0]})
    X = tds[["date"]]
    y = tds.target

    search_task = SearchTask("", endpoint=url)

    def initial_max_hit_rate() -> Optional[float]:
        return 1.0

    search_task.initial_max_hit_rate_v2 = initial_max_hit_rate
    search_keys = {"date": SearchKey.DATE}
    enricher = FeaturesEnricher(search_keys=search_keys, endpoint=url, logs_enabled=False)
    enricher.X = X
    enricher.y = y
    enricher._search_task = search_task
    enricher._FeaturesEnricher__cached_sampled_datasets = (X, y, X, dict(), search_keys)

    with pytest.raises(ValidationError, match=bundle.get("metrics_unfitted_enricher")):
        enricher.calculate_metrics()


def test_handle_index_search_keys(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)

    tds = pd.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "feature": [1, 2, 3],
        }
    )
    tds.set_index("date", inplace=True)
    tds["date"] = [date(2021, 1, 1), date(2021, 2, 1), date(2021, 3, 1)]
    search_keys = {"date": SearchKey.DATE}
    enricher = FeaturesEnricher(search_keys=search_keys, endpoint=url, logs_enabled=False)
    handled = enricher._FeaturesEnricher__handle_index_search_keys(tds, search_keys)  # type: ignore
    expected = pd.DataFrame({"feature": [1, 2, 3], "date": [date(2021, 1, 1), date(2021, 2, 1), date(2021, 3, 1)]})
    assert_frame_equal(handled, expected)


def test_correct_target_regression(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)

    tds = pd.DataFrame(
        {
            "date": [date(2020, 1, 1)] * 20,
            "target": [str(i) for i in range(1, 20)] + ["non_numeric_value"],
        }
    )
    enricher = FeaturesEnricher(search_keys={"date": SearchKey.DATE}, endpoint=url, logs_enabled=False)
    handled = enricher._FeaturesEnricher__correct_target(tds)  # type: ignore
    expected = pd.DataFrame({"date": [date(2020, 1, 1)] * 20, "target": [float(i) for i in range(1, 20)] + [np.nan]})
    assert_frame_equal(handled, expected)


def test_correct_target_multiclass(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)

    tds = pd.DataFrame(
        {
            "date": [date(2020, 1, 1)] * 10,
            "target": ["1", "2", "1", "2", "3", "single non numeric", "5", "6", "non numeric", "non numeric"],
        }
    )
    enricher = FeaturesEnricher(search_keys={"date": SearchKey.DATE}, endpoint=url, logs_enabled=False)
    handled = enricher._FeaturesEnricher__correct_target(tds)  # type: ignore
    print(handled)
    expected = pd.DataFrame(
        {
            "date": [date(2020, 1, 1)] * 10,
            "target": ["1", "2", "1", "2", np.nan, np.nan, np.nan, np.nan, "non numeric", "non numeric"],
        }
    )
    assert_frame_equal(handled, expected)


def test_correct_order_of_enriched_X(requests_mock: Mocker):
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df = df.sample(frac=1).reset_index(drop=True)
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    print("Train features")
    print(train_features)
    train_target = train_df["target"]
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].reset_index(drop=True)
    eval2_df = df[11000:12000].reset_index(drop=True)
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"].reset_index(drop=True)
    eval_set = [(eval1_features, eval1_target), (eval2_features, eval2_target)]

    search_keys = {"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE}
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        logs_enabled=False,
    )

    enricher.fit(train_features, train_target, eval_set=eval_set, calculate_metrics=False)

    df_with_eval_set_index = train_features.copy()
    df_with_eval_set_index["eval_set_index"] = 0
    for idx, eval_pair in enumerate(eval_set):
        eval_x, _ = eval_pair
        eval_df_with_index = eval_x.copy()
        eval_df_with_index["eval_set_index"] = idx + 1
        df_with_eval_set_index = pd.concat([df_with_eval_set_index, eval_df_with_index])

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )

    mock_features = pd.read_parquet(path_to_mock_features)
    converter = DateTimeSearchKeyConverter("rep_date")
    df_with_eval_set_index_with_date = converter.convert(df_with_eval_set_index)
    mock_features["system_record_id"] = pd.util.hash_pandas_object(
        df_with_eval_set_index_with_date[sorted(search_keys.keys())].reset_index(drop=True), index=False
    ).astype("Float64")
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, mock_features)

    enriched_df_with_eval_set = enricher.transform(df_with_eval_set_index)

    enriched_X = enriched_df_with_eval_set[enriched_df_with_eval_set.eval_set_index == 0]
    enriched_eval_X_1 = enriched_df_with_eval_set[enriched_df_with_eval_set.eval_set_index == 1]
    enriched_eval_X_2 = enriched_df_with_eval_set[enriched_df_with_eval_set.eval_set_index == 2]

    print("Enriched X")
    print(enriched_X)

    assert not enriched_X["feature"].isna().any()
    assert not enriched_eval_X_1["feature"].isna().any()
    assert not enriched_eval_X_2["feature"].isna().any()

    assert_frame_equal(train_features, enriched_X[train_features.columns])

    assert_frame_equal(eval1_features, enriched_eval_X_1[eval1_features.columns])

    assert_frame_equal(eval2_features, enriched_eval_X_2[eval2_features.columns])


def test_features_enricher_with_datetime(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)

    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="datetime_time_sin_1", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.001
                ),
                FeaturesMetadataV2(
                    name="datetime_time_sin_2", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.001
                ),
                FeaturesMetadataV2(
                    name="datetime_time_sin_24", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.001
                ),
                FeaturesMetadataV2(
                    name="datetime_time_sin_48", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.001
                ),
                FeaturesMetadataV2(
                    name="datetime_time_cos_1", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.001
                ),
                FeaturesMetadataV2(
                    name="datetime_time_cos_2", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.001
                ),
                FeaturesMetadataV2(
                    name="datetime_time_cos_24", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.001
                ),
                FeaturesMetadataV2(
                    name="datetime_time_cos_48", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.001
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data_with_time.parquet")
    df = pd.read_parquet(path)
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].reset_index(drop=True)
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"]

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        cv=CVType.time_series,
        logs_enabled=False,
    )

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
    )
    assert enriched_train_features.shape == (10000, 11)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 1
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1
    assert len(first_feature_info["Value preview"]) > 0 and len(first_feature_info["Value preview"]) < 30
    # Client features are no longer shown
    # assert enricher.features_info.loc[1, feature_name_header] == "datetime_time_sin_1"
    # assert enricher.features_info.loc[1, shap_value_header] == 0.001
    # assert enricher.features_info.loc[2, feature_name_header] == "datetime_time_sin_2"
    # assert enricher.features_info.loc[2, shap_value_header] == 0.001
    # assert enricher.features_info.loc[3, feature_name_header] == "datetime_time_sin_24"
    # assert enricher.features_info.loc[3, shap_value_header] == 0.001
    # assert enricher.features_info.loc[4, feature_name_header] == "datetime_time_sin_48"
    # assert enricher.features_info.loc[4, shap_value_header] == 0.001
    # assert enricher.features_info.loc[5, feature_name_header] == "datetime_time_cos_1"
    # assert enricher.features_info.loc[5, shap_value_header] == 0.001
    # assert enricher.features_info.loc[6, feature_name_header] == "datetime_time_cos_2"
    # assert enricher.features_info.loc[6, shap_value_header] == 0.001
    # assert enricher.features_info.loc[7, feature_name_header] == "datetime_time_cos_24"
    # assert enricher.features_info.loc[7, shap_value_header] == 0.001
    # assert enricher.features_info.loc[8, feature_name_header] == "datetime_time_cos_48"
    # assert enricher.features_info.loc[8, shap_value_header] == 0.001

    metrics = enricher.calculate_metrics()
    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment, eval_1_segment, eval_2_segment],
            rows_header: [10000, 1000, 1000],
            target_mean_header: [0.5044, 0.487, 0.486],
            baseline_gini: [-0.024345, 0.005368, 0.010041],
            enriched_gini: [-0.022148, 0.019638, -0.030032],
            uplift: [0.002198, 0.014270, -0.040073],
        }
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)


def test_idempotent_order_with_balanced_dataset(requests_mock: Mocker):
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)
    df = df[df["phone_num"] >= 10_000_000]

    search_keys = {"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE}
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        logs_enabled=False,
    )

    result_wrapper = DataFrameWrapper()

    def mocked_initial_search(self, trace_id, file_path, metadata, metrics, search_customization):
        result_wrapper.df = pd.read_parquet(file_path)
        raise TestException()

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    expected_result_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/expected_prepared.parquet"
    )

    expected_result_df = pd.read_parquet(expected_result_path).sort_values(by="system_record_id").reset_index(drop=True)

    def test(n_shuffles: int):
        train_df = df.head(10000)
        for _ in range(n_shuffles):
            train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_features = train_df.drop(columns="target")
        train_target = train_df["target"]
        eval1_df = df[10000:11000]
        for _ in range(n_shuffles):
            eval1_df = eval1_df.sample(frac=1).reset_index(drop=True)
        eval1_features = eval1_df.drop(columns="target")
        eval1_target = eval1_df["target"]
        eval2_df = df[11000:12000]
        for _ in range(n_shuffles):
            eval2_df = eval2_df.sample(frac=1).reset_index(drop=True)
        eval2_features = eval2_df.drop(columns="target")
        eval2_target = eval2_df["target"]
        eval_set = [(eval1_features, eval1_target), (eval2_features, eval2_target)]

        try:
            enricher.fit(train_features, train_target, eval_set, calculate_metrics=False)
        except TestException:
            pass

        actual_result_df = result_wrapper.df.sort_values(by="system_record_id").reset_index(drop=True)
        assert_frame_equal(actual_result_df, expected_result_df)

    try:
        for i in range(5):
            test(i)
    finally:
        _RestClient.initial_search_v2 = original_initial_search


def test_imbalanced_dataset(requests_mock: Mocker):
    pd.set_option("display.max_columns", 1000)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_initial_progress(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    ads_search_task_id = mock_initial_and_validation_summary(
        requests_mock,
        url,
        search_task_id,
        validation_search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)

    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="ads_feature", type="numeric", source="ads", hit_rate=100.0, shap_value=0.9),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=8000, max_hit_count=8000, hit_rate=1.0, hit_rate_percent=100.0
            ),
            eval_set_metrics=[],
        ),
    )
    path_to_mock_features = os.path.join(base_dir, "test_data/binary/features_imbalanced.parquet")

    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)
    mock_validation_raw_features(
        requests_mock, url, validation_search_task_id, path_to_mock_features, metrics_calculation=True
    )

    train_path = os.path.join(base_dir, "test_data/binary/initial_train_imbalanced.parquet")
    train_df = pd.read_parquet(train_path)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]

    default_min_sample_threshold = Dataset.MIN_SAMPLE_THRESHOLD
    Dataset.MIN_SAMPLE_THRESHOLD = 7_000

    search_keys = {"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE}
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        logs_enabled=False,
    )

    try:
        enricher.fit(train_features, train_target, calculate_metrics=False)

        metrics = enricher.calculate_metrics()

        print(metrics)

        assert metrics.loc[0, "Rows"] == 8000
        assert metrics.loc[0, "Mean target"] == 0.125
        assert metrics.loc[0, "Enriched GINI"] == 0.0
    finally:
        Dataset.MIN_SAMPLE_THRESHOLD = default_min_sample_threshold


def test_idempotent_order_with_imbalanced_dataset(requests_mock: Mocker):
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(base_dir, "test_data/binary/initial_train_imbalanced.parquet")
    eval1_path = os.path.join(base_dir, "test_data/binary/initial_eval1_imbalanced.parquet")
    eval2_path = os.path.join(base_dir, "test_data/binary/initial_eval2_imbalanced.parquet")
    initial_train_df = pd.read_parquet(train_path)

    initial_eval1_df = pd.read_parquet(eval1_path)
    initial_eval2_df = pd.read_parquet(eval2_path)

    default_min_sample_threshold = Dataset.MIN_SAMPLE_THRESHOLD
    Dataset.MIN_SAMPLE_THRESHOLD = 7_000

    search_keys = {"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE}
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        logs_enabled=False,
    )

    result_wrapper = DataFrameWrapper()

    def mocked_initial_search(self, trace_id, file_path, metadata, metrics, search_customization):
        result_wrapper.df = pd.read_parquet(file_path)
        raise TestException()

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    try:
        expected_result_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data/binary/expected_prepared_imbalanced.parquet"
        )

        expected_result_df = (
            pd.read_parquet(expected_result_path).sort_values(by="system_record_id").reset_index(drop=True)
        )

        def test(n_shuffles: int):
            train_df = initial_train_df.copy()
            for _ in range(n_shuffles):
                train_df = initial_train_df.sample(frac=1).reset_index(drop=True)
            train_features = train_df.drop(columns="target")
            train_target = train_df["target"]
            eval1_df = initial_eval1_df.copy()
            for _ in range(n_shuffles):
                eval1_df = eval1_df.sample(frac=1).reset_index(drop=True)
            eval1_features = eval1_df.drop(columns="target")
            eval1_target = eval1_df["target"]
            eval2_df = initial_eval2_df.copy()
            for _ in range(n_shuffles):
                eval2_df = eval2_df.sample(frac=1).reset_index(drop=True)
            eval2_features = eval2_df.drop(columns="target")
            eval2_target = eval2_df["target"]
            eval_set = [(eval1_features, eval1_target), (eval2_features, eval2_target)]

            try:
                enricher.fit(train_features, train_target, eval_set, calculate_metrics=False)
            except TestException:
                pass

            actual_result_df = result_wrapper.df.sort_values(by="system_record_id").reset_index(drop=True)

            assert_frame_equal(actual_result_df, expected_result_df)

        for i in range(5):
            print(f"Run {i} iteration")
            test(i)
    finally:
        _RestClient.initial_search_v2 = original_initial_search
        Dataset.MIN_SAMPLE_THRESHOLD = default_min_sample_threshold


def test_email_search_key(requests_mock: Mocker):
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    enricher = FeaturesEnricher(
        search_keys={"email": SearchKey.EMAIL},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    df = pd.DataFrame({"email": ["test1@gmail.com", "test2@mail.com", "test3@yahoo.com"], "target": [0, 1, 0]})
    original_search = Dataset.search
    original_min_count = Dataset.MIN_ROWS_COUNT

    def mock_search(
        self,
        *args,
        **kwargs,
    ):
        self.validate()
        columns = self.columns.to_list()
        assert set(columns) == {
            "system_record_id",
            "target",
            "hashed_email_64ff8c",
            "email_one_domain_3b0a68",
            "email_domain_10c73f",
        }
        assert {"hashed_email_64ff8c", "email_one_domain_3b0a68"} == {
            sk for sublist in self.search_keys for sk in sublist
        }
        raise TestException()

    Dataset.search = mock_search
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher.fit(df.drop(columns="target"), df.target)
        raise AssertionError("Search should fail")
    except AssertionError:
        raise
    except TestException:
        pass
    finally:
        Dataset.search = original_search
        Dataset.MIN_ROWS_COUNT = original_min_count


def test_composit_index_search_key(requests_mock: Mocker):
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    enricher = FeaturesEnricher(
        search_keys={"country": SearchKey.COUNTRY, "postal_code": SearchKey.POSTAL_CODE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    df = pd.DataFrame(
        {"country": ["GB", "EC", "US"], "postal_code": ["103305", "504938", "293049"], "target": [0, 1, 0]}
    )
    df.set_index(["country", "postal_code"])
    original_search = Dataset.search
    original_min_count = Dataset.MIN_ROWS_COUNT

    def mock_search(
        self,
        *args,
        **kwargs,
    ):
        self.validate()
        assert set(self.columns.to_list()) == {"system_record_id", "country_aff64e", "postal_code_13534a", "target"}
        assert "country_aff64e" in self.columns
        assert "postal_code_13534a"
        assert {"country_aff64e", "postal_code_13534a"} == {sk for sublist in self.search_keys for sk in sublist}
        # assert "country_fake_a" in self.columns
        # assert "postal_code_fake_a" in self.columns
        # assert {"country_fake_a", "postal_code_fake_a"} == {sk for sublist in self.search_keys for sk in sublist}
        raise TestException()

    Dataset.search = mock_search
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher.fit(df.drop(columns="target"), df.target)
        raise AssertionError("Search sould fail")
    except AssertionError:
        raise
    except TestException:
        pass
    finally:
        Dataset.search = original_search
        Dataset.MIN_ROWS_COUNT = original_min_count


def test_search_keys_autodetection(requests_mock: Mocker):
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    search_task_id = "123"
    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_initial_progress(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    ads_search_task_id = mock_initial_and_validation_summary(
        requests_mock,
        url,
        search_task_id,
        validation_search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="numeric", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    enricher = FeaturesEnricher(
        search_keys={"date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    df = pd.DataFrame(
        {
            # "country": ["EC", "GB", "US"],
            "postal_code": ["103305", "504938", "293049"],
            "phone": ["9992223311", "28376283746", "283764736"],
            "eml": ["test@mail.ru", "test2@gmail.com", "test3@yahoo.com"],
            "date": ["2021-01-01", "2022-01-01", "2023-01-01"],
            "target": [0, 1, 0],
        }
    )
    original_search = Dataset.search
    original_min_count = Dataset.MIN_ROWS_COUNT

    def mock_search(
        self,
        *args,
        **kwargs,
    ):
        self.validate()
        columns = set(self.columns.to_list())
        assert columns == {
            "system_record_id",
            "postal_code_13534a",
            "phone_45569d",
            "date_0e8763",
            "target",
            "hashed_email_64ff8c",
            "email_one_domain_3b0a68",
            "email_domain_10c73f",
        }
        assert {
            "postal_code_13534a",
            "phone_45569d",
            "hashed_email_64ff8c",
            "email_one_domain_3b0a68",
            "date_0e8763",
        } == {sk for sublist in self.search_keys for sk in sublist}
        search_task = SearchTask(search_task_id, self, endpoint=url, api_key="fake_api_key")
        search_task.provider_metadata_v2 = [
            ProviderTaskMetadataV2(
                features=[
                    FeaturesMetadataV2(name="feature", type="numeric", source="ads", hit_rate=100, shap_value=0.5)
                ]
            )
        ]
        return search_task

    Dataset.search = mock_search
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher.fit(df.drop(columns="target"), df.target)
    except AssertionError:
        raise
    except Exception as e:
        assert e.args[0].path == "/public/api/v2/search/123/progress"
    finally:
        Dataset.search = original_search
        Dataset.MIN_ROWS_COUNT = original_min_count

    df["country"] = "GB"
    # enricher.search_id = search_task_id

    old_validation = Dataset.validation

    def mock_validation(
        self,
        *args,
        **kwargs,
    ):
        self.validate(validate_target=False)
        assert {
            # "country_aff64e",
            "postal_code_13534a",
            "phone_45569d",
            "hashed_email_64ff8c",
            "email_one_domain_3b0a68",
            "date_0e8763",
        } == {sk for sublist in self.search_keys for sk in sublist}
        raise TestException()

    Dataset.validation = mock_validation

    try:
        enricher.transform(df.drop(columns="target"))
        raise AssertionError("Should fail")
    except TestException:
        pass
    finally:
        Dataset.validation = old_validation


def test_numbers_with_comma(requests_mock: Mocker):
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    enricher = FeaturesEnricher(
        search_keys={"date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    df = pd.DataFrame(
        {
            "date": ["2021-01-01", "2022-01-01", "2023-01-01"],
            "feature": ["12,5", "34,2", "45,7"],
            "target": [0, 1, 0],
        }
    )
    original_search = Dataset.search
    original_min_rows = Dataset.MIN_ROWS_COUNT

    def mock_search(
        self,
        *args,
        **kwargs,
    ):
        self.validate()
        assert self.data["feature_2ad562"].dtype == "float64"
        raise TestException()

    Dataset.search = mock_search
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher.fit(df.drop(columns="target"), df.target)
    except AssertionError:
        raise
    except TestException:
        pass
    finally:
        Dataset.search = original_search
        Dataset.MIN_ROWS_COUNT = original_min_rows


def test_diff_target_dups(requests_mock: Mocker):
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    enricher = FeaturesEnricher(
        search_keys={"date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    df = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-01", "2023-01-01", "2023-01-01"],
            "feature": [11, 11, 12, 13],
            "target": [0, 1, 0, 1],
        }
    )
    original_search = Dataset.search
    original_min_rows = Dataset.MIN_ROWS_COUNT

    def mock_search(
        self,
        *args,
        **kwargs,
    ):
        self.validate()
        assert len(self.data) == 2
        print(self.data)
        assert self.data.loc[0, "date_0e8763"] == 1672531200000
        assert self.data.loc[0, "feature_2ad562"] == 12
        assert self.data.loc[0, "target"] == 0
        assert self.data.loc[1, "date_0e8763"] == 1672531200000
        assert self.data.loc[1, "feature_2ad562"] == 13
        assert self.data.loc[1, "target"] == 1
        return SearchTask("123", self, endpoint=url, api_key="fake_api_key")

    Dataset.search = mock_search
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher.fit(df.drop(columns="target"), df.target)
        raise AssertionError("Search should fail")
    except AssertionError:
        raise
    except Exception as e:
        assert e.args[0].path == "/public/api/v2/search/123/progress"
    finally:
        Dataset.search = original_search
        Dataset.MIN_ROWS_COUNT = original_min_rows


def test_unsupported_arguments(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_initial_progress(requests_mock, url, search_task_id)
    mock_validation_progress(requests_mock, url, validation_search_task_id)
    ads_search_task_id = mock_initial_and_validation_summary(
        requests_mock,
        url,
        search_task_id,
        validation_search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1)],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
                ModelEvalSet(
                    eval_set_index=2,
                    hit_rate=0.99,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=990, hit_rate=0.99, hit_rate_percent=99.0
                    ),
                ),
            ],
        ),
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    enricher = FeaturesEnricher(
        search_keys={"date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        unsupported_argument="some_value",
    )

    df = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-02", "2023-01-01", "2023-01-02"],
            "feature": [11, 10, 12, 13],
            "target": [0, 1, 0, 1],
        }
    )

    original_min_rows = Dataset.MIN_ROWS_COUNT
    Dataset.MIN_ROWS_COUNT = 3
    try:
        # with pytest.raises(NoMockAddress):
        enricher.fit(
            df.drop(columns="target"),
            df["target"],
            [(df.drop(columns="target"), df["target"])],
            "unsupported_positional_argument",
            unsupported_key_argument=False,
        )

        # with pytest.raises(NoMockAddress):
        enricher.fit_transform(
            df.drop(columns="target"),
            df["target"],
            [(df.drop(columns="target"), df["target"])],
            "unsupported_positional_argument",
            unsupported_key_argument=False,
        )

        enricher.transform(df.drop(columns="target"), "unsupported_positional_argument", unsupported_key_argument=False)

        enricher.calculate_metrics(
            df.drop(columns="target"),
            df["target"],
            [(df.drop(columns="target"), df["target"])],
            "unsupported_positional_argument",
            unsupported_key_argument=False,
        )
    finally:
        Dataset.MIN_ROWS_COUNT = original_min_rows


class DataFrameWrapper:
    def __init__(self):
        self.df = None


class TestException(Exception):
    def __init__(self):
        super().__init__()
