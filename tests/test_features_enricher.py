import os
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from requests_mock.mocker import Mocker

from upgini import FeaturesEnricher, SearchKey
from upgini.errors import ValidationError
from upgini.metadata import (
    CVType,
    FeaturesMetadataV2,
    HitRateMetrics,
    ModelEvalSet,
    ProviderTaskMetadataV2,
    RuntimeParameters,
)
from upgini.resource_bundle import bundle
from upgini.search_task import SearchTask
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter

from .utils import (
    mock_default_requests,
    mock_get_features_meta,
    mock_get_metadata,
    mock_get_task_metadata_v2,
    mock_initial_search,
    mock_initial_summary,
    mock_raw_features,
    mock_validation_raw_features,
    mock_validation_search,
    mock_validation_summary,
)

train_segment = bundle.get("quality_metrics_train_segment")
eval_1_segment = bundle.get("quality_metrics_eval_segment").format(1)
eval_2_segment = bundle.get("quality_metrics_eval_segment").format(2)
match_rate_header = bundle.get("quality_metrics_match_rate_header")
rows_header = bundle.get("quality_metrics_rows_header")
baseline_rocauc = bundle.get("quality_metrics_baseline_header").format("roc_auc")
enriched_rocauc = bundle.get("quality_metrics_enriched_header").format("roc_auc")
uplift = bundle.get("quality_metrics_uplift_header")
feature_name_header = bundle.get("features_info_name")
shap_value_header = bundle.get("features_info_shap")
hitrate_header = bundle.get("features_info_hitrate")


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
    pd.set_option("mode.chained_assignment", "raise")
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "feature", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[],
    )
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
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
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
    assert enriched_train_features.shape == (10000, 4)

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
    )
    assert enriched_train_features.shape == (10000, 4)

    metrics = enricher.calculate_metrics()
    expected_metrics = (
        pd.DataFrame(
            {
                "segment": [train_segment, eval_1_segment, eval_2_segment],
                rows_header: [10000, 1000, 1000],
                enriched_rocauc: [0.486751, 0.507267, 0.528008],
            }
        )
        .set_index("segment")
        .rename_axis("")
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
    pd.set_option("mode.chained_assignment", "raise")
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "feature", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[],
    )
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
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
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
    assert enriched_train_features.shape == (10000, 4)

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        keep_input=True,
        calculate_metrics=False,
    )
    assert enriched_train_features.shape == (10000, 4)

    metrics = enricher.calculate_metrics()
    expected_metrics = (
        pd.DataFrame(
            {
                "segment": [train_segment, eval_1_segment, eval_2_segment],
                rows_header: [10000, 1000, 1000],
                baseline_rocauc: [0.529017, 0.490646, 0.523306],
                enriched_rocauc: [0.510232, 0.492119, 0.520055],
                uplift: [-0.018785, 0.001472, -0.003252],
            }
        )
        .set_index("segment")
        .rename_axis("")
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
    pd.set_option("mode.chained_assignment", "raise")
    pd.set_option("display.max_columns", 1000)
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
    pd.set_option("mode.chained_assignment", "raise")
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "feature", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[],
    )
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
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
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
        search_keys={1: SearchKey.PHONE, 2: SearchKey.DATE},
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
    assert enriched_train_features.shape == (10000, 4)

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
    )
    assert enriched_train_features.shape == (10000, 4)

    metrics = enricher.calculate_metrics()
    expected_metrics = (
        pd.DataFrame(
            {
                "segment": [train_segment, eval_1_segment, eval_2_segment],
                rows_header: [10000, 1000, 1000],
                enriched_rocauc: [0.486751, 0.507267, 0.528008],
            }
        )
        .set_index("segment")
        .rename_axis("")
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
    pd.set_option("mode.chained_assignment", "raise")
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "feature", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[],
    )
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
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
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
    assert enriched_train_features.shape == (10000, 4)
    assert enriched_train_features.index.name == "custom_index_name"

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
    )
    assert enriched_train_features.shape == (10000, 4)

    metrics = enricher.calculate_metrics()
    expected_metrics = (
        pd.DataFrame(
            {
                "segment": [train_segment, eval_1_segment, eval_2_segment],
                rows_header: [10000, 1000, 1000],
                enriched_rocauc: [0.486751, 0.507267, 0.528008],
            }
        )
        .set_index("segment")
        .rename_axis("")
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
    pd.set_option("mode.chained_assignment", "raise")
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
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
        ),
    )
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/complex_feature_name_features.parquet"
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

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
    expected_metrics = (
        pd.DataFrame(
            {
                "segment": [train_segment],
                rows_header: [5319],
                # match_rate_header: [99.0],
                baseline_rocauc: [0.501952],
                enriched_rocauc: [0.504399],
                uplift: [0.002448],
            }
        )
        .set_index("segment")
        .rename_axis("")
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["f_feature123"]
    assert enricher.feature_importances_ == [0.9]
    assert len(enricher.features_info) == 2
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "f_feature123"
    assert first_feature_info[shap_value_header] == 0.9
    assert first_feature_info[hitrate_header] == 99.0
    second_feature_info = enricher.features_info.iloc[1]
    assert second_feature_info[feature_name_header] == "cos(3,freq=W-SUN)"
    assert second_feature_info[shap_value_header] == 0.1
    assert second_feature_info[hitrate_header] == 100.0


def test_features_enricher_fit_transform_runtime_parameters(requests_mock: Mocker):
    pd.set_option("mode.chained_assignment", "raise")
    url = "http://fake_url2"
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "feature", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[{"name": "SystemRecordId_473310000", "importance": 1.0, "matchedInPercent": 100.0}],
    )
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
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
        ],
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


def test_search_with_only_personal_keys(requests_mock: Mocker):
    url = "https://some.fake.url"

    mock_default_requests(requests_mock, url)

    with pytest.raises(Exception):
        FeaturesEnricher(
            search_keys={"phone": SearchKey.PHONE, "email": SearchKey.EMAIL}, endpoint=url, logs_enabled=False
        )


def test_filter_by_importance(requests_mock: Mocker):
    url = "https://some.fake.url"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)

    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "feature", "importance": 0.7, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[],
    )
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

    # assert enricher.enriched_X is not None
    # assert len(enricher.enriched_X) == 10000
    # assert enricher.enriched_X.columns.to_list() == ["SystemRecordId_473310000", "phone_num", "rep_date"]
    # assert enricher.enriched_eval_set is not None
    # assert len(enricher.enriched_eval_set) == 2000
    # assert enricher.enriched_eval_set.columns.to_list() == [
    #     "SystemRecordId_473310000",
    #     "phone_num",
    #     "rep_date",
    #     "eval_set_index"
    # ]

    metrics = enricher.calculate_metrics(importance_threshold=0.8)

    # expected_metrics = (
    #     pd.DataFrame(
    #         {
    #             "segment": [train_segment, eval_1_segment, eval_2_segment],
    #             rows_header: [10000, 1000, 1000],
    #             baseline_rocauc: [0.5, 0.5, 0.5],
    #         }
    #     )
    #     .set_index("segment")
    #     .rename_axis("")
    # )
    # print("Expected metrics: ")
    # print(expected_metrics)
    # print("Actual metrics: ")
    # print(metrics)

    assert metrics is None
    # assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
        ],
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

    assert train_features.shape == (10000, 3)

    test_features = enricher.transform(eval1_features, keep_input=True, importance_threshold=0.8)

    assert test_features.shape == (1000, 3)


def test_filter_by_max_features(requests_mock: Mocker):
    url = "https://some.fake.url"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)

    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "feature", "importance": 0.7, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[],
    )
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
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
        ],
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    train_features = enricher.fit_transform(
        train_features, train_target, eval_set=eval_set, calculate_metrics=False, keep_input=True, max_features=0
    )

    assert train_features.shape == (10000, 3)

    test_features = enricher.transform(eval1_features, keep_input=True, max_features=0)

    assert test_features.shape == (1000, 3)


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

    assert enricher.calculate_metrics() is None


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
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    # mock_get_features_meta(
    #     requests_mock,
    #     url,
    #     ads_search_task_id,
    #     ads_features=[{"name": "feature", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
    #     etalon_features=[],
    # )
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

    enricher.fit(
        train_features,
        train_target,
        eval_set=eval_set,
        calculate_metrics=False
    )

    df_with_eval_set_index = train_features.copy()
    df_with_eval_set_index["eval_set_index"] = 0
    for idx, eval_pair in enumerate(eval_set):
        eval_x, _ = eval_pair
        eval_df_with_index = eval_x.copy()
        eval_df_with_index["eval_set_index"] = idx + 1
        df_with_eval_set_index = pd.concat([df_with_eval_set_index, eval_df_with_index])

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )

    mock_features = pd.read_parquet(path_to_mock_features)
    converter = DateTimeSearchKeyConverter("rep_date")
    df_with_eval_set_index_with_date = converter.convert(df_with_eval_set_index)
    mock_features["system_record_id"] = [
        hash(tuple(row)) for row in df_with_eval_set_index_with_date[sorted(search_keys.keys())].values
    ]
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
    pd.set_option("mode.chained_assignment", "raise")
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.parquet"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
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
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.9,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data_with_time.parquet")
    df = pd.read_parquet(path)
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
    assert enriched_train_features.shape == (10000, 12)

    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        keep_input=True,
        calculate_metrics=False,
    )
    assert enriched_train_features.shape == (10000, 12)

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 9
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1
    assert enricher.features_info.loc[1, feature_name_header] == "datetime_time_sin_1"
    assert enricher.features_info.loc[1, shap_value_header] == 0.001
    assert enricher.features_info.loc[2, feature_name_header] == "datetime_time_sin_2"
    assert enricher.features_info.loc[2, shap_value_header] == 0.001
    assert enricher.features_info.loc[3, feature_name_header] == "datetime_time_sin_24"
    assert enricher.features_info.loc[3, shap_value_header] == 0.001
    assert enricher.features_info.loc[4, feature_name_header] == "datetime_time_sin_48"
    assert enricher.features_info.loc[4, shap_value_header] == 0.001
    assert enricher.features_info.loc[5, feature_name_header] == "datetime_time_cos_1"
    assert enricher.features_info.loc[5, shap_value_header] == 0.001
    assert enricher.features_info.loc[6, feature_name_header] == "datetime_time_cos_2"
    assert enricher.features_info.loc[6, shap_value_header] == 0.001
    assert enricher.features_info.loc[7, feature_name_header] == "datetime_time_cos_24"
    assert enricher.features_info.loc[7, shap_value_header] == 0.001
    assert enricher.features_info.loc[8, feature_name_header] == "datetime_time_cos_48"
    assert enricher.features_info.loc[8, shap_value_header] == 0.001

    metrics = enricher.calculate_metrics()
    expected_metrics = (
        pd.DataFrame(
            {
                "segment": [train_segment, eval_1_segment, eval_2_segment],
                rows_header: [10000, 1000, 1000],
                baseline_rocauc: [0.495165, 0.498326, 0.462597],
                enriched_rocauc: [0.498229, 0.512327, 0.474131],
                uplift: [0.003064, 0.014001, 0.011534],
            }
        )
        .set_index("segment")
        .rename_axis("")
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)
