import logging
import os
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from requests_mock import NoMockAddress
from requests_mock.mocker import Mocker

from upgini.dataset import Dataset
from upgini.errors import ValidationError
from upgini.features_enricher import FeaturesEnricher, hash_input
from upgini.http import _RestClient
from upgini.metadata import (
    SYSTEM_RECORD_ID,
    TARGET,
    CVType,
    FeaturesMetadataV2,
    HitRateMetrics,
    ModelEvalSet,
    ModelTaskType,
    ProviderTaskMetadataV2,
    RuntimeParameters,
    SearchKey,
)
from upgini.normalizer.normalize_utils import Normalizer
from upgini.resource_bundle import bundle
from upgini.search_task import SearchTask
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter
from upgini.utils.sample_utils import SampleConfig

from .utils import (
    mock_default_requests,
    mock_get_metadata,
    mock_get_selected_features,
    mock_get_task_metadata_v2,
    mock_initial_and_validation_summary,
    mock_initial_progress,
    mock_initial_search,
    mock_initial_summary,
    mock_raw_features,
    mock_set_selected_features,
    mock_validation_progress,
    mock_validation_raw_features,
    mock_validation_search,
    mock_validation_summary,
)

feature_name_header = bundle.get("features_info_name")
shap_value_header = bundle.get("features_info_shap")

SearchTask.PROTECT_FROM_RATE_LIMIT = False
SearchTask.POLLING_DELAY_SECONDS = 0
pd.set_option("mode.chained_assignment", "raise")
pd.set_option("display.max_columns", 1000)
FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_data/enricher/",
)


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


def test_features_enricher(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features_validation = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
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
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns=["SystemRecordId_473310000"], inplace=True)
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
        select_features=False,
    )
    assert enriched_train_features.shape == (10000, 6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_features_info.csv"), index=False
        )

    expected_features_info = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_features_info.csv")
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    metrics = enricher.calculate_metrics()

    if update_metrics_flag:
        metrics.to_csv(os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher.csv"))

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_features_info_after_metrics.csv"),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_features_info_after_metrics.csv")
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    # Check that renaming of columns doesn't change the metrics
    train_features.rename(columns={"client_feature": "клиентская фича"}, inplace=True)
    eval1_features.rename(columns={"client_feature": "клиентская фича"}, inplace=True)
    eval2_features.rename(columns={"client_feature": "клиентская фича"}, inplace=True)
    enriched_train_features = enricher.fit_transform(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
        calculate_metrics=False,
        keep_input=True,
        select_features=False,
    )

    metrics = enricher.calculate_metrics()
    logging.warning(metrics)
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)


def test_eval_set_with_diff_order_of_columns(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(requests_mock, url, search_task_id)
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
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
    eval1_features = eval1_features[list(eval1_features.columns)]
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
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
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
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns=["SystemRecordId_473310000", "client_feature"], inplace=True)
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
            select_features=False,
            keep_input=True,
        )
        assert enriched_train_features.shape == (6, 5)

        enriched_train_features = enricher.fit_transform(
            train_features,
            train_target,
            calculate_metrics=False,
            select_features=False,
            keep_input=False,
        )
        assert enriched_train_features.shape == (6, 3)

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


def test_saved_features_enricher(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
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
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 2,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="numeric", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="numeric", source="etalon", hit_rate=100.0, shap_value=0.4
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="numeric",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="numeric",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature", "client_feature"])
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features, metrics_calculation=True)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features)
    mock_validation_raw_features(
        requests_mock, url, validation_search_task_id, path_to_mock_features, metrics_calculation=True
    )

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    train_df = df.head(10000)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"].copy()
    eval1_df = df[10000:11000].reset_index(drop=True)
    eval1_features = eval1_df.drop(columns="target")
    eval1_target = eval1_df["target"].reset_index(drop=True).copy()
    eval2_df = df[11000:12000]
    eval2_features = eval2_df.drop(columns="target")
    eval2_target = eval2_df["target"].copy()

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
    logging.warning(enriched_train_features)
    assert enriched_train_features.shape == (10000, 4)

    # Check keep_input=False
    enriched_train_features = enricher.transform(
        train_features, keep_input=False
    )
    assert enriched_train_features.shape == (10000, 1)

    df_for_transform = train_features.copy()
    df_for_transform["some_feature_on_transform"] = np.random.randint(0, 1000, size=len(train_features))

    # Check keep_input=True
    enriched_train_features = enricher.transform(
        df_for_transform, keep_input=True
    )
    assert enriched_train_features.shape == (10000, 6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_saved_features_enricher_features_info.csv"),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_saved_features_enricher_features_info.csv")
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    mock_set_selected_features(requests_mock, url, search_task_id, ["feature", "client_feature"])

    metrics = enricher.calculate_metrics(
        train_features,
        train_target,
        eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)],
    )

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_saved_features_enricher.csv"),
            index=False,
        )

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_features_enricher/test_saved_features_enricher.csv"))

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR, "test_features_enricher/test_saved_features_enricher_features_info_after_metrics.csv"
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_saved_features_enricher_features_info_after_metrics.csv")
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    # Check imbalanced target metrics
    random = np.random.RandomState(42)
    train_random_indices = random.choice(train_target.index, size=9000, replace=False)
    train_target.loc[train_random_indices] = 0

    metrics = enricher.calculate_metrics(train_features, train_target)

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_saved_features_enricher_imbalanced_target.csv"),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_saved_features_enricher_imbalanced_target.csv")
    )

    assert metrics is not None
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)


def test_features_enricher_with_demo_key(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features_validation = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
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
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns=["SystemRecordId_473310000", "client_feature"], inplace=True)
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
        select_features=False,
    )
    assert enriched_train_features.shape == (10000, 5)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_demo_key_features_info.csv"),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_demo_key_features_info.csv")
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    metrics = enricher.calculate_metrics()

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_demo_key_metrics.csv"),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_demo_key_metrics.csv")
    )

    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR,
                "test_features_enricher/test_features_enricher_with_demo_key_features_info_after_metrics.csv",
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(
            FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_demo_key_features_info_after_metrics.csv"
        )
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()


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


def test_features_enricher_with_numpy(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features_validation = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
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
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns=["SystemRecordId_473310000", "client_feature"], inplace=True)
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
        select_features=False,
    )
    assert enriched_train_features.shape == (10000, 5)

    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_numpy_features_info.csv"),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_numpy_features_info.csv")
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    metrics = enricher.calculate_metrics()
    assert metrics is not None

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_numpy_metrics.csv"),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_numpy_metrics.csv")
    )

    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_numpy_features_info_after_metrics.csv"
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(
            FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_numpy_features_info_after_metrics.csv"
        )
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    enricher.transform(train_features)


def test_features_enricher_with_named_index(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features_validation = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
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
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns=["SystemRecordId_473310000", "client_feature"], inplace=True)
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
        select_features=False,
    )
    assert enriched_train_features.shape == (10000, 5)
    assert enriched_train_features.index.name == "custom_index_name"

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_named_index_features_info.csv"
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_named_index_features_info.csv")
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    metrics = enricher.calculate_metrics()

    assert metrics is not None

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_named_index_metrics.csv"),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_named_index_metrics.csv")
    )

    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR,
                "test_features_enricher/test_features_enricher_with_named_index_features_info_after_metrics.csv",
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(
            FIXTURE_DIR,
            "test_features_enricher/test_features_enricher_with_named_index_features_info_after_metrics.csv",
        )
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()


def test_features_enricher_with_index_column(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features_validation = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
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
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns=["SystemRecordId_473310000", "client_feature"], inplace=True)
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
        select_features=False,
    )
    assert enriched_train_features.shape == (10000, 5)
    assert "index" not in enriched_train_features.columns

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_index_column_features_info.csv"
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_index_column_features_info.csv")
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    metrics = enricher.calculate_metrics()
    assert metrics is not None

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_index_column_metrics.csv"),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_index_column_metrics.csv")
    )

    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR,
                "test_features_enricher/test_features_enricher_with_index_column_features_info_after_metrics.csv",
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(
            FIXTURE_DIR,
            "test_features_enricher/test_features_enricher_with_index_column_features_info_after_metrics.csv",
        )
    )
    expected_features_info["Updates"] = expected_features_info["Updates"].astype("string")
    enricher.features_info["Updates"] = enricher.features_info["Updates"].astype("string")

    assert_frame_equal(expected_features_info, enricher.features_info, atol=1e-6)

    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()


def test_features_enricher_with_complex_feature_names(requests_mock: Mocker, update_metrics_flag: bool):
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
                    "name": "cos_3_freq_w_sun__0a6bf9",
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
                    name="cos_3_freq_w_sun__0a6bf9", type="numerical", source="etalon", hit_rate=100.0, shap_value=0.1
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=5319, max_hit_count=5266, hit_rate=0.99, hit_rate_percent=99.0
            ),
            features_used_for_embeddings=["cos_3_freq_w_sun__0a6bf9"],
        ),
    )
    mock_get_selected_features(requests_mock, url, search_task_id, ["f_feature123", "cos(3,freq=W-SUN)"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["f_feature123", "cos(3,freq=W-SUN)"])
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/complex_feature_name_features_with_entity_system_record_id.parquet",
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

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR,
                "test_features_enricher/test_features_enricher_with_complex_feature_names_features_info.csv",
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(
            FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_complex_feature_names_features_info.csv"
        )
    )
    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

    metrics = enricher.calculate_metrics()
    assert metrics is not None
    features_info = enricher.get_features_info()
    assert features_info is not None

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(
                FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_complex_feature_names_metrics.csv"
            ),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(
            FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_complex_feature_names_metrics.csv"
        )
    )

    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

    if update_metrics_flag:
        enricher.features_info.to_csv(
            os.path.join(
                FIXTURE_DIR,
                "test_features_enricher/"
                "test_features_enricher_with_complex_feature_names_features_info_after_metrics.csv",
            ),
            index=False,
        )

    expected_features_info = pd.read_csv(
        os.path.join(
            FIXTURE_DIR,
            "test_features_enricher/test_features_enricher_with_complex_feature_names_features_info_after_metrics.csv",
        )
    )
    assert enricher.feature_names_ == expected_features_info["Feature name"].tolist()
    assert enricher.feature_importances_ == expected_features_info["SHAP value"].tolist()

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
        assert "cos_3_freq_w_sun__0a6bf9" in self.data.columns
        assert runtime_parameters.properties["features_for_embeddings"] == "cos_3_freq_w_sun__0a6bf9"
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
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features_validation = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
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
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=1.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=1.0,
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
    mock_get_selected_features(
        requests_mock,
        url,
        search_task_id,
        ["feature", "datetime_day_in_quarter_sin", "datetime_day_in_quarter_cos"],
    )
    mock_set_selected_features(
        requests_mock,
        url,
        search_task_id,
        ["feature", "datetime_day_in_quarter_sin", "datetime_day_in_quarter_cos"],
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
        stability_threshold=10,
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
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)

    transformed = enricher.transform(train_features)

    transform_req = None
    transform_url = url + "/public/api/v2/search/validation?initialSearchTaskId=" + search_task_id
    for elem in requests_mock.request_history:
        if elem.url == transform_url:
            transform_req = elem

    assert transform_req is not None
    assert "runtimeProperty1" in str(transform_req.body)
    assert "runtimeValue1" in str(transform_req.body)

    assert transformed.shape == (10000, 5)


def test_features_enricher_fit_custom_loss(requests_mock: Mocker):
    url = "http://fake_url2"
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
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


def test_validation_metrics_calculation(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)

    tds = pd.DataFrame({"date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)], "target": [0, 1, 0]})
    X = tds[["date"]]
    y = tds.target

    search_task = SearchTask("")

    def initial_max_hit_rate() -> Optional[float]:
        return 1.0

    search_task.initial_max_hit_rate_v2 = initial_max_hit_rate
    search_keys = {"date": SearchKey.DATE}
    enricher = FeaturesEnricher(search_keys=search_keys, endpoint=url, logs_enabled=False)
    enricher.X = X
    enricher.y = y
    enricher._search_task = search_task
    datasets_hash = hash_input(X, y, (X, y))
    enricher._FeaturesEnricher__cached_sampled_datasets[datasets_hash] = (X, y, X, dict(), search_keys, [])

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
    handled = enricher._FeaturesEnricher__handle_index_search_keys(tds, search_keys)
    enricher._FeaturesEnricher__prepare_search_keys(handled, search_keys, False)
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
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.5
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
        ),
    )
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature", "client_feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature", "client_feature"])
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

    enricher.fit(train_features, train_target, eval_set=eval_set, calculate_metrics=False, stability_agg_func="min")

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
    search_keys_copy = search_keys.copy()
    normalizer = Normalizer()
    df_with_eval_set_index_with_date, search_keys_copy, converter.generated_features = normalizer.normalize(
        df_with_eval_set_index_with_date, search_keys_copy, converter.generated_features
    )
    mock_features["system_record_id"] = pd.util.hash_pandas_object(
        df_with_eval_set_index_with_date[sorted(search_keys_copy.keys())].reset_index(drop=True), index=False
    ).astype("float64")
    mock_features["entity_system_record_id"] = mock_features["system_record_id"]
    mock_features = mock_features.drop_duplicates(subset=["entity_system_record_id"], keep="first")
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, mock_features)

    enriched_df_with_eval_set = enricher.transform(df_with_eval_set_index)

    enriched_X = enriched_df_with_eval_set[enriched_df_with_eval_set.eval_set_index == 0]
    enriched_eval_X_1 = enriched_df_with_eval_set[enriched_df_with_eval_set.eval_set_index == 1]
    enriched_eval_X_2 = enriched_df_with_eval_set[enriched_df_with_eval_set.eval_set_index == 2]

    print("Enriched X")
    print(enriched_X)

    assert "feature" in enriched_X.columns
    assert not enriched_X["feature"].isna().any()
    assert not enriched_eval_X_1["feature"].isna().any()
    assert not enriched_eval_X_2["feature"].isna().any()

    assert_frame_equal(train_features, enriched_X[train_features.columns])

    assert_frame_equal(eval1_features, enriched_eval_X_1[eval1_features.columns])

    assert_frame_equal(eval2_features, enriched_eval_X_2[eval2_features.columns])


def test_features_enricher_with_datetime(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features_validation = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
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
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.001,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.001,
                ),
                FeaturesMetadataV2(
                    name="datetime_second_sin_60_d4ab6d",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.001,
                ),
                FeaturesMetadataV2(
                    name="datetime_second_cos_60_be3be7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.001,
                ),
                FeaturesMetadataV2(
                    name="datetime_minute_sin_60_609cc6",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.001,
                ),
                FeaturesMetadataV2(
                    name="datetime_minute_cos_60_c8c837",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.001,
                ),
                FeaturesMetadataV2(
                    name="datetime_minute_sin_30_73fdef",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.001,
                ),
                FeaturesMetadataV2(
                    name="datetime_minute_cos_30_256cfb",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.001,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
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
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)

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
        select_features=False,
        stability_threshold=0.2,
    )
    assert enriched_train_features.shape == (10000, 11)

    logging.warning(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 1
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info[feature_name_header] == "feature"
    assert first_feature_info[shap_value_header] == 10.1
    assert len(first_feature_info["Value preview"]) > 0 and len(first_feature_info["Value preview"]) < 30

    metrics = enricher.calculate_metrics()
    assert metrics is not None

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_datetime_metrics.csv"),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_datetime_metrics.csv")
    )

    assert_frame_equal(expected_metrics, metrics, atol=1e-6)


def test_idempotent_order_with_balanced_dataset(requests_mock: Mocker, update_metrics_flag: bool):
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns=["SystemRecordId_473310000", "client_feature"], inplace=True)
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
        raise TestException

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    expected_result_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/expected_prepared_with_entity_system_record_id.parquet",
    )

    expected_result_df = pd.read_parquet(expected_result_path).sort_values(by="system_record_id").reset_index(drop=True)

    def test(n_shuffles: int, expected_df: pd.DataFrame):
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
        if update_metrics_flag:
            actual_result_df.to_parquet(expected_result_path)
            expected_df = actual_result_df
        assert_frame_equal(actual_result_df, expected_df)

    try:
        for i in range(5):
            test(i, expected_result_df)
    finally:
        _RestClient.initial_search_v2 = original_initial_search


def test_imbalanced_dataset(requests_mock: Mocker, update_metrics_flag: bool):
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["ads_feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["ads_feature"])
    path_to_mock_features = os.path.join(
        base_dir, "test_data/binary/features_imbalanced_with_entity_system_record_id.parquet"
    )

    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    path_to_mock_features_validation = os.path.join(
        base_dir, "test_data/binary/features_imbalanced_with_entity_system_record_id_validation.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)
    mock_validation_raw_features(
        requests_mock, url, validation_search_task_id, path_to_mock_features_validation, metrics_calculation=True
    )

    train_path = os.path.join(base_dir, "test_data/binary/initial_train_imbalanced.parquet")
    train_df = pd.read_parquet(train_path)
    train_features = train_df.drop(columns="target")
    train_target = train_df["target"]

    search_keys = {"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE}
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        logs_enabled=False,
        sample_config=SampleConfig(binary_min_sample_threshold=7_000),
    )

    enricher.fit(train_features, train_target, calculate_metrics=False)

    metrics = enricher.calculate_metrics()
    assert metrics is not None

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(
                FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_imbalanced_dataset_metrics.csv"
            ),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_imbalanced_dataset_metrics.csv")
    )
    assert_frame_equal(expected_metrics, metrics, atol=1e-6)


def test_idempotent_order_with_imbalanced_dataset(requests_mock: Mocker, update_metrics_flag: bool):
    pd.set_option("display.max_columns", 1000)
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(base_dir, "test_data/binary/initial_train_imbalanced.parquet")
    eval1_path = os.path.join(base_dir, "test_data/binary/initial_eval1_imbalanced.parquet")
    eval2_path = os.path.join(base_dir, "test_data/binary/initial_eval2_imbalanced.parquet")
    initial_train_df = pd.read_parquet(train_path)

    initial_eval1_df = pd.read_parquet(eval1_path)
    initial_eval1_df = initial_eval1_df[
        ~initial_eval1_df.set_index(["phone_num", "rep_date", "target"]).index.isin(
            initial_train_df.set_index(["phone_num", "rep_date", "target"]).index
        )
    ]
    initial_eval2_df = pd.read_parquet(eval2_path)
    initial_eval2_df = initial_eval2_df[
        ~initial_eval2_df.set_index(["phone_num", "rep_date", "target"]).index.isin(
            initial_train_df.set_index(["phone_num", "rep_date", "target"]).index
        )
    ]

    search_keys = {"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE}
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        logs_enabled=False,
        sample_config=SampleConfig(binary_min_sample_threshold=1000),
    )

    result_wrapper = DataFrameWrapper()

    def mocked_initial_search(self, trace_id, file_path, metadata, metrics, search_customization):
        result_wrapper.df = pd.read_parquet(file_path)
        raise TestException

    original_initial_search = _RestClient.initial_search_v2
    _RestClient.initial_search_v2 = mocked_initial_search

    try:
        expected_result_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data/binary/expected_prepared_imbalanced.parquet"
        )

        expected_result_df = (
            pd.read_parquet(expected_result_path).sort_values(by="system_record_id").reset_index(drop=True)
        )
        expected_result_df["phone_num_a54a33"] = expected_result_df["phone_num_a54a33"].astype("Int64")
        expected_result_df["rep_date_f5d6bb"] = expected_result_df["rep_date_f5d6bb"].astype("Int64")

        def test(n_shuffles: int, expected_df: pd.DataFrame):
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
            actual_result_df["phone_num_a54a33"] = actual_result_df["phone_num_a54a33"].astype("Int64")
            actual_result_df["rep_date_f5d6bb"] = actual_result_df["rep_date_f5d6bb"].astype("Int64")

            if update_metrics_flag:
                actual_result_df.to_parquet(expected_result_path)
                expected_df = actual_result_df
            assert_frame_equal(actual_result_df, expected_df)

        for i in range(5):
            print(f"Run {i} iteration")
            test(i, expected_result_df)
    finally:
        _RestClient.initial_search_v2 = original_initial_search


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
            "entity_system_record_id",
            "target",
            "email_822444",
            "email_822444_hem",
            "email_822444_one_domain",
            "email_domain_10c73f",
            "current_date_b993c4",
        }
        assert {"email_822444", "email_822444_hem", "email_822444_one_domain", "current_date_b993c4"} == {
            sk for sublist in self.search_keys for sk in sublist
        }
        raise TestException

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
        assert set(self.columns.to_list()) == {
            "system_record_id",
            "entity_system_record_id",
            "country_aff64e",
            "postal_code_13534a",
            "current_date_b993c4",
            "target",
        }
        assert "country_aff64e" in self.columns
        assert "postal_code_13534a"
        assert {"country_aff64e", "postal_code_13534a", "current_date_b993c4"} == {
            sk for sublist in self.search_keys for sk in sublist
        }
        # assert "country_fake_a" in self.columns
        # assert "postal_code_fake_a" in self.columns
        # assert {"country_fake_a", "postal_code_fake_a"} == {sk for sublist in self.search_keys for sk in sublist}
        raise TestException

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
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_sin_65d4f7",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
                ),
                FeaturesMetadataV2(
                    name="datetime_day_in_quarter_cos_eeb97a",
                    type="NUMERIC",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.0,
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
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
            "entity_system_record_id",
            "postal_code_13534a",
            "phone_45569d",
            "date_0e8763",
            "target",
            "eml_13033c",
            "eml_13033c_hem",
            "eml_13033c_one_domain",
            "eml_domain_7e1966",
        }
        assert {
            "postal_code_13534a",
            "phone_45569d",
            "eml_13033c",
            "eml_13033c_hem",
            "eml_13033c_one_domain",
            "date_0e8763",
        } == {sk for sublist in self.search_keys for sk in sublist}
        search_task = SearchTask(search_task_id, self, rest_client=enricher.rest_client)
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
        if hasattr(e.args[0], "path"):
            assert e.args[0].path == "/public/api/v2/search/123/progress"
        else:
            raise e
    finally:
        Dataset.search = original_search
        Dataset.MIN_ROWS_COUNT = original_min_count

    assert enricher._get_group_columns(df, enricher.fit_search_keys) == ["phone_45569d"]

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
            "eml_13033c",
            "eml_13033c_hem",
            "eml_13033c_one_domain",
            "date_0e8763",
        } == {sk for sublist in self.search_keys for sk in sublist}
        raise TestException

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
        raise TestException

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
        logging.warning(self.data)
        assert self.data.loc[0, "date_0e8763"] == 1672531200000
        assert self.data.loc[0, "feature_2ad562"] == 13
        assert self.data.loc[0, "target"] == 1
        assert self.data.loc[1, "date_0e8763"] == 1672531200000
        assert self.data.loc[1, "feature_2ad562"] == 12
        assert self.data.loc[1, "target"] == 0
        return SearchTask("123", self, rest_client=enricher.rest_client)

    Dataset.search = mock_search
    Dataset.MIN_ROWS_COUNT = 1

    try:
        enricher.fit(df.drop(columns="target"), df.target)
        raise AssertionError("Search should fail")
    except AssertionError:
        raise
    except Exception as e:
        if hasattr(e.args[0], "path"):
            assert e.args[0].path == "/public/api/v2/search/123/progress"
        else:
            raise e
    finally:
        Dataset.search = original_search
        Dataset.MIN_ROWS_COUNT = original_min_rows


def test_unsupported_arguments(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
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
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=100.0, shap_value=0.1
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature", "client_feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature", "client_feature"])
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
            "date": ["2021-01-01", "2021-01-02", "2023-01-01", "2023-01-02", "2023-01-03"],
            "client_feature": [11, 10, 12, 13, 14],
            "target": [0, 1, 0, 1, 0],
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

        with pytest.raises(NoMockAddress):
            enricher.calculate_metrics(
                df.drop(columns="target"),
                df["target"],
                [(df.drop(columns="target"), df["target"])],
                "unsupported_positional_argument",
                unsupported_key_argument=False,
            )
    finally:
        Dataset.MIN_ROWS_COUNT = original_min_rows


def test_multikey_metrics_without_external_features():
    # TODO test case when there is no external features found and we have datetime + email
    # that produce "client" features and multiple email produce multiple email_domain features
    pass


def test_select_features(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/binary/mock_features_with_entity_system_record_id.parquet",
    )

    if pd.__version__ >= "2.2.0":
        print("Use features for pandas 2.2")
        features_file = "validation_features_v3_with_entity_system_record_id.parquet"
    elif pd.__version__ >= "2.0.0":
        print("Use features for pandas 2.0")
        features_file = "validation_features_v2_with_entity_system_record_id.parquet"
    else:
        print("Use features for pandas 1.*")
        features_file = "validation_features_v1_with_entity_system_record_id.parquet"
    path_to_mock_features_validation = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"test_data/binary/{features_file}"
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
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
            {
                "index": 1,
                "name": "phone_num_a54a33",
                "originalName": "phone_num",
                "dataType": "STRING",
                "meaningType": "MSISDN",
            },
            {
                "index": 2,
                "name": "rep_date_f5d6bb",
                "originalName": "rep_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {
                "index": 3,
                "name": "client_feature_8ddf40",
                "originalName": "client_feature",
                "dataType": "INT",
                "meaningType": "FEATURE",
            },
            {
                "index": 4,
                "name": "datetime_day_in_quarter_sin_65d4f7",
                "originalName": "datetime_day_in_quarter_sin_65d4f7",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
            {
                "index": 5,
                "name": "datetime_day_in_quarter_cos_eeb97a",
                "originalName": "datetime_day_in_quarter_cos_eeb97a",
                "dataType": "DECIMAL",
                "meaningType": "FEATURE",
            },
        ],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(name="feature", type="NUMERIC", source="ads", hit_rate=99.0, shap_value=10.1),
                FeaturesMetadataV2(
                    name="client_feature_8ddf40", type="NUMERIC", source="etalon", hit_rate=99.0, shap_value=0.0
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
    mock_get_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_set_selected_features(requests_mock, url, search_task_id, ["feature"])
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_features_validation)

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
        select_features=False,
    )
    assert enriched_train_features.shape == (10000, 6)

    metrics = enricher.calculate_metrics()
    assert metrics is not None

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_select_features_metrics.csv"),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_select_features_metrics.csv")
    )

    assert_frame_equal(expected_metrics, metrics, atol=1e-6)

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
        select_features=True,
    )
    assert enriched_train_features.shape == (10000, 3)
    assert "client_feature" not in enriched_train_features.columns

    metrics = enricher.calculate_metrics()
    assert metrics is not None

    if update_metrics_flag:
        metrics.to_csv(
            os.path.join(
                FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_select_features_metrics_2.csv"
            ),
            index=False,
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_features_enricher/test_features_enricher_with_select_features_metrics_2.csv")
    )

    assert_frame_equal(expected_metrics, metrics, atol=1e-6)


def test_id_columns_validation(requests_mock: Mocker):
    url = "http://fake_url2"

    mock_default_requests(requests_mock, url)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path, sep=",")
    df.drop(columns="SystemRecordId_473310000", inplace=True)

    enricher = FeaturesEnricher(
        search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        cv=CVType.time_series,
        logs_enabled=False,
        id_columns=["unexistent_column"],
    )

    with pytest.raises(ValidationError, match="Id column unexistent_column not found in X"):
        enricher.fit(
            df.drop(columns="target"),
            df.target,
        )


def test_adjusting_cv():
    # TODO
    assert True
    # enricher = FeaturesEnricher(
    #     search_keys={"phone_num": SearchKey.PHONE, "rep_date": SearchKey.DATE},
    #     endpoint=url,
    #     api_key="fake_api_key",
    #     date_format="%Y-%m-%d",
    #     cv=CVType.time_series,
    #     logs_enabled=False,
    # )
    # enricher._FeaturesEnricher__adjust_cv(df)


def test_eval_x_intersection_with_x():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    intersecting_eval_set_X = pd.DataFrame({"a": [3, 4, 5], "b": [6, 7, 8]})
    non_intersecting_eval_set_X = pd.DataFrame({"a": [10, 11, 12], "b": [13, 14, 15]})
    eval_set_y = pd.Series([1, 0, 1])
    enricher = FeaturesEnricher(search_keys={"a": SearchKey.CUSTOM_KEY})

    enricher._validate_eval_set_pair(X, (non_intersecting_eval_set_X, eval_set_y))

    with pytest.raises(ValidationError, match="Eval set X has rows that are present in train set X"):
        enricher._validate_eval_set_pair(X, (intersecting_eval_set_X, eval_set_y))


def test_add_fit_system_record_id():
    df = pd.DataFrame(
        {
            "index": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "date_renamed": [
                "2021-01-03",
                "2021-01-02",
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
                "2021-01-01",
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
            ],  # noqa: E501
            "phone_renamed": [4, 5, 6, 7, 8, 9, 10, 11, 12],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0],
            "eval_set_index": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    )
    df.set_index("index", inplace=True)
    fit_search_keys = {"date_renamed": SearchKey.DATE, "phone_renamed": SearchKey.PHONE}
    fit_columns_renaming = {"date_renamed": "date", "phone_renamed": "phone"}
    id_columns = ["phone_renamed"]
    cv = None  # CVType.time_series
    model_task_type = ModelTaskType.BINARY
    logger = logging.getLogger(__name__)
    df = FeaturesEnricher._add_fit_system_record_id(
        df,
        fit_search_keys,
        SYSTEM_RECORD_ID,
        TARGET,
        fit_columns_renaming,
        id_columns,
        cv,
        model_task_type,
        logger,
    )

    expected_df = pd.DataFrame(
        {
            "index": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "system_record_id": [2, 1, 0, 4, 5, 3, 6, 7, 8],
            "date_renamed": [
                "2021-01-03",
                "2021-01-02",
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
                "2021-01-01",
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
            ],  # noqa: E501
            "phone_renamed": [4, 5, 6, 7, 8, 9, 10, 11, 12],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0],
            "eval_set_index": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    )
    expected_df.set_index("index", inplace=True)

    assert_frame_equal(df, expected_df)


class DataFrameWrapper:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None


class TestException(Exception):
    pass
