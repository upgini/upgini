import datetime
import json
import os

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.testing import assert_frame_equal
from requests_mock.mocker import Mocker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit

from upgini.errors import ValidationError
from upgini.features_enricher import FeaturesEnricher, hash_input
from upgini.metadata import (
    CVType,
    FeaturesMetadataV2,
    HitRateMetrics,
    ModelEvalSet,
    ModelTaskType,
    ProviderTaskMetadataV2,
    SearchKey,
)
from upgini.metrics import (
    CATBOOST_BINARY_PARAMS,
    LIGHTGBM_BINARY_PARAMS,
    CatBoostWrapper,
    LightGBMWrapper,
    OtherEstimatorWrapper,
    _get_scorer_by_name,
)
from upgini.normalizer.normalize_utils import Normalizer
from upgini.resource_bundle import bundle
from upgini.search_task import SearchTask
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter

from .utils import (
    mock_default_requests,
    mock_get_metadata,
    mock_get_task_metadata_v2,
    mock_get_task_metadata_v2_from_file,
    mock_initial_progress,
    mock_initial_search,
    mock_initial_summary,
    mock_raw_features,
    mock_target_outliers,
    mock_target_outliers_file,
    mock_validation_progress,
    mock_validation_raw_features,
    mock_validation_search,
    mock_validation_summary,
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_data/enricher/",
)

SearchTask.PROTECT_FROM_RATE_LIMIT = False
SearchTask.POLLING_DELAY_SECONDS = 0


def test_real_case_metric_binary(requests_mock: Mocker, update_metrics_flag: bool):
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/",
    )

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
                    "name": "score",
                    "originalName": "score",
                    "dataType": "INT",
                    "meaningType": "FEATURE",
                },
                {
                    "index": 1,
                    "name": "request_date",
                    "originalName": "request_date",
                    "dataType": "INT",
                    "meaningType": "DATE",
                },
                {"index": 2, "name": "target", "originalName": "target", "dataType": "INT", "meaningType": "DATE"},
                {
                    "index": 3,
                    "name": "system_record_id",
                    "originalName": "system_record_id",
                    "dataType": "INT",
                    "meaningType": "SYSTEM_RECORD_ID",
                },
            ],
            "searchKeys": [["rep_date"]],
            "hierarchicalGroupKeys": [],
            "hierarchicalSubgroupKeys": [],
            "rowsCount": 30505,
        },
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(
                    name="score",
                    type="numeric",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.368092,
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=10000, hit_rate=1.0, hit_rate_percent=100.0
            ),
            eval_set_metrics=[
                ModelEvalSet(
                    eval_set_index=1,
                    hit_rate=1.0,
                    hit_rate_metrics=HitRateMetrics(
                        etalon_row_count=1000, max_hit_count=1000, hit_rate=1.0, hit_rate_percent=100.0
                    ),
                ),
            ],
        ),
    )
    path_to_mock_features = os.path.join(BASE_DIR, "real_train_df.parquet")
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    # train = pd.read_parquet(os.path.join(BASE_DIR, "real_train.parquet"))
    # train.sort_index()
    # X = train[["request_date", "score"]]
    # y = train["target1"].rename("target")
    # test = pd.read_parquet(os.path.join(BASE_DIR, "real_test.parquet"))
    # eval_set = [(test[["request_date", "score"]], test["target1"].rename("target"))]

    search_keys = {"request_date": SearchKey.DATE}
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        country_code="RU",
        search_id=search_task_id,
        logs_enabled=False,
    )

    train = pd.read_parquet(os.path.join(BASE_DIR, "real_train_df.parquet"))
    enricher.X = train.drop(columns=["system_record_id", "target1"])
    enricher.y = train["target1"]

    test = pd.read_parquet(os.path.join(BASE_DIR, "real_test_df.parquet"))
    test = test[
        ~test.set_index(["request_date", "target1"]).index.isin(train.set_index(["request_date", "target1"]).index)
    ]
    enricher.eval_set = [(test.drop(columns=["system_record_id", "target1"]), test["target1"])]

    enriched_X = train.drop(columns="target1")
    sampled_X = enriched_X
    sampled_y = train["target1"]

    enriched_eval_X = test.drop(columns="target1")
    sampled_eval_X = enriched_eval_X
    sampled_eval_y = test["target1"]

    columns_renaming = {c: c for c in enriched_X.columns}

    datasets_hash = hash_input(enricher.X, enricher.y, enricher.eval_set)
    enricher._FeaturesEnricher__cached_sampled_datasets[datasets_hash] = (
        sampled_X,
        sampled_y,
        enriched_X,
        {0: (sampled_eval_X, enriched_eval_X, sampled_eval_y)},
        search_keys,
        columns_renaming,
    )

    metrics = enricher.calculate_metrics()

    if update_metrics_flag:
        metrics.to_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_real_case_metric_binary.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_real_case_metric_binary.csv"))
    assert_frame_equal(expected_metrics, metrics)


def test_demo_metrics(requests_mock: Mocker, update_metrics_flag: bool):
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/demo/",
    )

    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    with open(os.path.join(BASE_DIR, "file_meta.json")) as f:
        file_meta = json.load(f)
    requests_mock.get(url + f"/public/api/v2/search/{search_task_id}/metadata", json=file_meta)
    with open(os.path.join(BASE_DIR, "provider_meta.json"), "rb") as f:
        provider_meta_json = json.load(f)
        provider_meta = ProviderTaskMetadataV2.parse_obj(provider_meta_json)
    mock_get_task_metadata_v2(requests_mock, url, ads_search_task_id, provider_meta)
    mock_raw_features(requests_mock, url, search_task_id, os.path.join(BASE_DIR, "x_enriched.parquet"))

    search_keys = {"country": SearchKey.COUNTRY, "Postal_code": SearchKey.POSTAL_CODE}
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        generate_features=["combined", "company_txt"],
        search_id=search_task_id,
        logs_enabled=False,
    )

    x_sampled = pd.read_parquet(os.path.join(BASE_DIR, "x_sampled.parquet"))
    y_sampled = pd.read_parquet(os.path.join(BASE_DIR, "y_sampled.parquet"))["target"]
    enriched_X = pd.read_parquet(os.path.join(BASE_DIR, "x_enriched.parquet"))

    enricher.X = x_sampled.drop(columns="system_record_id")
    enricher.y = y_sampled
    enricher.eval_set = []

    columns_renaming = {c: c for c in x_sampled.columns}

    datasets_hash = hash_input(enricher.X, enricher.y)
    enricher._FeaturesEnricher__cached_sampled_datasets[datasets_hash] = (
        x_sampled,
        y_sampled,
        enriched_X,
        dict(),
        search_keys,
        columns_renaming,
    )
    enricher.fit_select_features = False

    metrics = enricher.calculate_metrics(scoring="mean_absolute_error")

    if update_metrics_flag:
        metrics.to_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_demo_metrics.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_demo_metrics.csv"))
    assert_frame_equal(expected_metrics, metrics)


def test_default_metric_binary(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="etalon",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat_508476", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input.csv"))
    df["feature_2_cat"] = np.random.randint(0, 10, len(df))
    df["feature_2_cat"] = df["feature_2_cat"].astype("string").astype("category")
    df = df.reset_index().rename(columns={"index": "high_cardinality_feature"})
    df["date"] = pd.date_range(datetime.date(2020, 1, 1), periods=len(df))
    df_train = df[0:500]
    X = df_train[["phone", "date", "feature1", "high_cardinality_feature"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "date", "feature1", "high_cardinality_feature"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "date", "feature1", "high_cardinality_feature"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE, "date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    metrics_df = enricher.calculate_metrics()
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_binary.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_binary.csv"))
    assert_frame_equal(metrics_df, expected_metrics)


def test_default_metric_multiclass(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="etalon",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat_508476", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
                FeaturesMetadataV2(
                    name="postal_code_13534a", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.1
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input_multiclass.csv"))
    df["feature_2_cat"] = np.random.randint(0, 10, len(df))
    df["feature_2_cat"] = df["feature_2_cat"].astype("string").astype("category")
    df = df.reset_index().rename(columns={"index": "high_cardinality_feature"})
    df["date"] = pd.date_range(datetime.date(2020, 1, 1), periods=len(df))
    df_train = df[0:500]
    X = df_train[["phone", "date", "feature1", "high_cardinality_feature"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "date", "feature1", "high_cardinality_feature"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "date", "feature1", "high_cardinality_feature"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE, "date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    metrics_df = enricher.calculate_metrics()
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_multiclass.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_multiclass.csv"))
    assert_frame_equal(metrics_df, expected_metrics)


def test_default_metric_regression_with_outliers(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    path_to_metadata = os.path.join(FIXTURE_DIR, "metadata_regression_date_country_postal.json")
    mock_get_task_metadata_v2_from_file(requests_mock, url, ads_search_task_id, path_to_metadata)

    search_keys = {"date": SearchKey.DATE, "country": SearchKey.COUNTRY, "postal_code": SearchKey.POSTAL_CODE}
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_regression_date_country_postal.parquet")
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    df = pd.read_parquet(os.path.join(FIXTURE_DIR, "tds_regression_date_country_postal.parquet"))
    search_keys_copy = search_keys.copy()
    df_with_eval_set_index = df.copy()
    df_with_eval_set_index.loc[df_with_eval_set_index.segment == "train", "eval_set_index"] = 0
    df_with_eval_set_index.loc[df_with_eval_set_index.segment == "oot", "eval_set_index"] = 1
    df_with_eval_set_index.drop(columns="segment")

    mock_features = pd.read_parquet(path_to_mock_features)

    converter = DateTimeSearchKeyConverter("date")
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
    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)

    mock_validation_progress(requests_mock, url, validation_search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, mock_features)

    target_outliers_path = os.path.join(FIXTURE_DIR, "target_outliers_regression_date_country_postal.parquet")
    target_outliers_id = mock_target_outliers(requests_mock, url, search_task_id)
    mock_target_outliers_file(requests_mock, url, target_outliers_id, target_outliers_path)

    df_train = df.query("segment == 'train'").drop(columns="segment")
    eval_1 = df.query("segment == 'oot'").drop(columns="segment")
    X = df_train.drop(columns="target")
    y = df_train["target"]
    eval_X_1 = eval_1.drop(columns="target")
    eval_y_1 = eval_1["target"]
    eval_set = [(eval_X_1, eval_y_1)]
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False, select_features=False)

    assert len(enriched_X) == len(X)

    assert not enriched_X["f_weather_date_weather_umap_42_4ac1a34c"].isna().any()

    metrics_df = enricher.calculate_metrics()
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(
            os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_regression_with_outliers.csv"), index=False
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_regression_with_outliers.csv")
    )
    assert_frame_equal(metrics_df, expected_metrics)


def test_default_metric_binary_custom_loss(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat_508476", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input.csv"))
    df["feature_2_cat"] = np.random.randint(0, 10, len(df))
    df["feature_2_cat"] = df["feature_2_cat"].astype("string").astype("category")
    df["date"] = pd.date_range(datetime.date(2020, 1, 1), periods=len(df))
    df_train = df[0:500]
    X = df_train[["phone", "date", "feature1"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "date", "feature1"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "date", "feature1"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE, "date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        loss="binary",
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    metrics_df = enricher.calculate_metrics()
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(
            os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_binary_custom_loss.csv"), index=False
        )

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_binary_custom_loss.csv"))
    assert_frame_equal(metrics_df, expected_metrics)


def test_default_metric_binary_shuffled(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat_508476", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.0
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input.csv"))
    df["feature_2_cat"] = np.random.randint(0, 10, len(df))
    df["feature_2_cat"] = df["feature_2_cat"].astype("string").astype("category")
    df["date"] = pd.date_range(datetime.date(2020, 1, 1), periods=len(df))
    df_train = df[0:500]
    df_train = df_train.sample(frac=1)
    X = df_train[["phone", "date", "feature1"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_1 = eval_1.sample(frac=1)
    eval_2 = df[750:1000]
    eval_2 = eval_2.sample(frac=1)
    eval_X_1 = eval_1[["phone", "date", "feature1"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "date", "feature1"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE, "date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    metrics_df = enricher.calculate_metrics()
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(
            os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_binary_shuffled.csv"), index=False
        )

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_binary_shuffled.csv"))
    assert_frame_equal(metrics_df, expected_metrics)


def test_blocked_timeseries_rmsle(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    # TODO replace with input dataset with date search key and regression target
    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input.csv"))
    df_train = df[0:500]
    X = df_train[["phone", "feature1"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "feature1"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "feature1"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        cv=CVType.blocked_time_series,
        logs_enabled=False,
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    metrics_df = enricher.calculate_metrics(scoring="RMSLE", inner_call=True)
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_blocked_timeseries_rmsle.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_blocked_timeseries_rmsle.csv"))
    assert_frame_equal(metrics_df, expected_metrics)


# @pytest.mark.skip
def test_catboost_metric_binary(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_parquet(os.path.join(FIXTURE_DIR, "input.parquet"))
    df_train = df[0:500]
    X = df_train[["phone", "feature1"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "feature1"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "feature1"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE}, endpoint=url, api_key="fake_api_key", logs_enabled=False
    )

    with pytest.raises(ValidationError, match=bundle.get("metrics_unfitted_enricher")):
        enricher.calculate_metrics()

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    from catboost import CatBoostClassifier

    estimator = CatBoostClassifier(**CATBOOST_BINARY_PARAMS)
    metrics_df = enricher.calculate_metrics(estimator=estimator, scoring="roc_auc")
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_catboost_metric_binary.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_catboost_metric_binary.csv"))

    assert_frame_equal(metrics_df, expected_metrics, atol=10**-6)


# @pytest.mark.skip
def test_catboost_metric_binary_with_cat_features(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="cat_feature2_3ef372",
                    type="categorical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.2,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="country_aff64e",
                    type="categorical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_parquet(os.path.join(FIXTURE_DIR, "input_with_cat.parquet"))
    df_train = df[0:500]
    X = df_train[["phone", "country", "feature1", "cat_feature2"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "country", "feature1", "cat_feature2"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "country", "feature1", "cat_feature2"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE, "country": SearchKey.COUNTRY},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    with pytest.raises(ValidationError, match=bundle.get("metrics_unfitted_enricher")):
        enricher.calculate_metrics()

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False, select_features=False)

    assert len(enriched_X) == len(X)

    from catboost import CatBoostClassifier

    estimator = CatBoostClassifier(random_seed=42, verbose=False, cat_features=[1, 3])
    metrics_df = enricher.calculate_metrics(estimator=estimator, scoring="roc_auc")
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(
            os.path.join(FIXTURE_DIR, "test_metrics/test_catboost_metric_binary_with_cat_features.csv"), index=False
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_metrics/test_catboost_metric_binary_with_cat_features.csv")
    )

    assert_frame_equal(metrics_df, expected_metrics, atol=10**-6)


# @pytest.mark.skip
def test_lightgbm_metric_binary(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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

    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input.csv"))
    df_train = df[0:500]
    X = df_train[["phone", "feature1"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "feature1"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "feature1"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    #     with pytest.raises(ValidationError, match=r".*mean_absolute_persentage_error is not a valid scoring value.*"):
    with pytest.raises(ValidationError, match=r"mean_absolute_persentage_error is not a valid scoring value."):
        enricher.calculate_metrics(scoring="mean_absolute_persentage_error")

    assert len(enriched_X) == len(X)

    from lightgbm import LGBMClassifier  # type: ignore

    estimator = LGBMClassifier(random_seed=42)
    metrics_df = enricher.calculate_metrics(estimator=estimator, scoring="mean_absolute_error")
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_lightgbm_metric_binary.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_lightgbm_metric_binary.csv"))

    assert_frame_equal(metrics_df, expected_metrics, atol=10**-6)

    estimator2 = LGBMClassifier(random_seed=42, max_depth=5, n_estimators=10)
    metrics2 = enricher.calculate_metrics(estimator=estimator2, scoring="mean_absolute_error")
    assert metrics2 is not None

    assert not metrics2.equals(metrics_df), "Metrics should be different for different model parameters"


def test_lightgbm_metric_binary_with_cat_features(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="cat_feature2_3ef372",
                    type="categorical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.2,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="country_aff64e",
                    type="categorical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_parquet(os.path.join(FIXTURE_DIR, "input_with_cat.parquet"))
    df_train = df[0:500]
    X = df_train[["phone", "country", "feature1", "cat_feature2"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "country", "feature1", "cat_feature2"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "country", "feature1", "cat_feature2"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE, "country": SearchKey.COUNTRY},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    with pytest.raises(ValidationError, match=bundle.get("metrics_unfitted_enricher")):
        enricher.calculate_metrics()

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False, select_features=False)

    assert len(enriched_X) == len(X)

    estimator = LGBMClassifier(**LIGHTGBM_BINARY_PARAMS)

    metrics_df = enricher.calculate_metrics(estimator=estimator, scoring="roc_auc")
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(
            os.path.join(FIXTURE_DIR, "test_metrics/test_lightgbm_metric_binary_with_cat_features.csv"), index=False
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_metrics/test_lightgbm_metric_binary_with_cat_features.csv")
    )

    assert_frame_equal(metrics_df, expected_metrics, atol=10**-6)


def test_rf_metric_rmse(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input.csv"))
    df_train = df[0:500]
    X = df_train[["phone", "feature1"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "feature1"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "feature1"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE}, endpoint=url, api_key="fake_api_key", logs_enabled=False
    )

    with pytest.raises(ValidationError, match=bundle.get("metrics_unfitted_enricher")):
        enricher.calculate_metrics()

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    estimator = RandomForestClassifier(random_state=42)
    metrics_df = enricher.calculate_metrics(estimator=estimator, scoring="rmse")
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_rf_metric_rmse.csv"), index=False)

    expected_metrics = pd.read_csv(os.path.join(FIXTURE_DIR, "test_metrics/test_rf_metric_rmse.csv"))

    assert_frame_equal(metrics_df, expected_metrics, atol=10**-6)


def test_default_metric_binary_with_string_feature(requests_mock: Mocker, update_metrics_flag: bool):
    url = "http://fake_url2"
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
                    name="ads_feature1",
                    type="numerical",
                    source="ads",
                    hit_rate=99.0,
                    shap_value=10.1,
                ),
                FeaturesMetadataV2(
                    name="feature1_422b73",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat_508476", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.01
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9900, hit_rate=0.99, hit_rate_percent=99.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features_with_entity_system_record_id.parquet")
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
    path_to_mock_validation_features = os.path.join(
        FIXTURE_DIR, "validation_features_with_entity_system_record_id.parquet"
    )
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_parquet(os.path.join(FIXTURE_DIR, "input_with_string_feature.parquet"))
    df_train = df[0:500]
    X = df_train[["phone", "feature1", "feature_2_cat"]]
    y = df_train["target"]
    eval_1 = df[500:750]
    eval_2 = df[750:1000]
    eval_X_1 = eval_1[["phone", "feature1", "feature_2_cat"]]
    eval_y_1 = eval_1["target"]
    eval_X_2 = eval_2[["phone", "feature1", "feature_2_cat"]]
    eval_y_2 = eval_2["target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE}, endpoint=url, api_key="fake_api_key", logs_enabled=False
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    metrics_df = enricher.calculate_metrics()
    assert metrics_df is not None

    if update_metrics_flag:
        metrics_df.to_csv(
            os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_binary_with_string_feature.csv"), index=False
        )

    expected_metrics = pd.read_csv(
        os.path.join(FIXTURE_DIR, "test_metrics/test_default_metric_binary_with_string_feature.csv")
    )

    assert_frame_equal(metrics_df, expected_metrics, atol=10**-6)


def approx(value: float):
    return pytest.approx(value, abs=0.000001)


def test_catboost_wrapper_with_different_feature_types():
    df = pd.DataFrame(
        {
            "constant_feature": [[1], [1], [1], [1], [1]],
            "text_feature": ["a", "b", "b", "a", "b"],
            "cat_float": [1.5, 2.3, 3.1, 4.7, 5.2],
            "cat_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "cat_int": [1, 2, 3, 1, 2],
            "cat_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "cat_str_category": pd.Categorical(["a", "b", "c", "a", "b"]),
            "cat_str_object": ["a", "b", "c", "a", "b"],
            "cat_str": pd.Series(["a", "b", "c", "a", "b"], dtype="string"),
            "cat_bool": [True, False, True, False, True],
            "cat_bool_category": pd.Categorical([True, False, True, False, True]),
            "cat_mix": [1, "a", True, 1.5, "b"],
            "cat_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
            "num_float": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_int": [1, 2, 3, 1, 2],
            "num_str": pd.Series(["x", "y", "z", "x", "y"], dtype="string"),
            "num_str_object": ["x", "y", "z", "x", "y"],
            "num_bool": [True, False, True, False, True],
            "num_mix": [1, "a", True, 1.5, "b"],
            "num_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "num_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "num_str_category": pd.Categorical(["x", "y", "z", "x", "y"]),
            "num_bool_category": pd.Categorical([True, False, True, False, True]),
            "num_mix_category": pd.Categorical([1, "a", True, 1.5, "b"]),
            "num_array_of_numbers": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
            "num_emb1": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb2": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb3": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb4": [1.2, 3.4, 5.6, 7.8, 9.0],
            "target": [0, 1, 0, 1, 0],
        }
    )

    scoring, metric_name, multiplier = _get_scorer_by_name("roc_auc")

    wrapper = CatBoostWrapper(
        CatBoostClassifier(verbose=False),
        scorer=scoring,
        cat_features=[
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
        ],
        text_features=["text_feature"],
        metric_name=metric_name,
        multiplier=multiplier,
        cv=KFold(n_splits=2),
        target_type=ModelTaskType.BINARY,
    )

    prepared_x, _, _, params = wrapper._prepare_to_fit(df.drop("target", axis=1), df["target"])
    assert set(prepared_x.columns.to_list()) == set(
        [
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "num_float",
            "num_int",
            "num_bool",
            "num_float_category",
            "num_int_category",
            "num_bool_category",
            "num_array_of_numbers",
            "__grouped_embeddings",
        ]
    )
    assert prepared_x["text_feature"].dtype == "object"
    assert prepared_x["cat_float"].dtype == "object"
    assert prepared_x["cat_float_category"].dtype == "object"
    assert prepared_x["cat_int"].dtype == "int64"
    assert prepared_x["cat_int_category"].dtype == "Int64"
    assert prepared_x["cat_str_category"].dtype == "category"
    assert prepared_x["cat_str_object"].dtype == "object"
    assert prepared_x["cat_str"].dtype == "object"
    assert prepared_x["cat_bool"].dtype == "Int64"
    assert prepared_x["cat_bool_category"].dtype == "Int64"
    assert prepared_x["cat_mix"].dtype == "object"
    assert prepared_x["cat_date"].dtype == "object"
    assert prepared_x["num_float"].dtype == "float64"
    assert prepared_x["num_int"].dtype == "int64"
    assert prepared_x["num_bool"].dtype == "Int64"
    assert prepared_x["num_float_category"].dtype == "float64"
    assert prepared_x["num_int_category"].dtype == "int64"
    assert prepared_x["num_bool_category"].dtype == "Int64"
    assert prepared_x["num_array_of_numbers"].dtype == "object"
    assert prepared_x["__grouped_embeddings"].dtype == "object"
    assert set(wrapper.cat_features) == set(
        [
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "cat_float",
            "cat_float_category",
            "cat_int",
        ]
    )
    assert set(wrapper.grouped_embedding_features) == set(["__grouped_embeddings", "num_array_of_numbers"])
    assert set(params["embedding_features"]) == set(["__grouped_embeddings", "num_array_of_numbers"])
    assert params["text_features"] == ["text_feature"]
    assert set(params["cat_features"]) == set(
        [
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "cat_float",
            "cat_float_category",
            "cat_int",
        ]
    )

    prepared_x2, _, params2 = wrapper._prepare_to_calculate(df.drop("target", axis=1), df["target"])
    assert_frame_equal(prepared_x, prepared_x2)
    assert params == params2

    result = wrapper.cross_val_predict(df.drop("target", axis=1), df["target"])
    assert result.get_display_metric() == "0.750 ± 0.250"


def test_catboost_wrapper_with_different_feature_types_time_series():
    df = pd.DataFrame(
        {
            "constant_feature": [[1], [1], [1], [1], [1]],
            "text_feature": ["a", "b", "b", "a", "b"],
            "cat_float": [1.5, 2.3, 3.1, 4.7, 5.2],
            "cat_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "cat_int": [1, 2, 3, 1, 2],
            "cat_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "cat_str_category": pd.Categorical(["a", "b", "c", "a", "b"]),
            "cat_str_object": ["a", "b", "c", "a", "b"],
            "cat_str": pd.Series(["a", "b", "c", "a", "b"], dtype="string"),
            "cat_bool": [True, False, True, False, True],
            "cat_bool_category": pd.Categorical([True, False, True, False, True]),
            "cat_mix": [1, "a", True, 1.5, "b"],
            "cat_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
            "num_float": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_int": [1, 2, 3, 1, 2],
            "num_str": pd.Series(["x", "y", "z", "x", "y"], dtype="string"),
            "num_str_object": ["x", "y", "z", "x", "y"],
            "num_bool": [True, False, True, False, True],
            "num_mix": [1, "a", True, 1.5, "b"],
            "num_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "num_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "num_str_category": pd.Categorical(["x", "y", "z", "x", "y"]),
            "num_bool_category": pd.Categorical([True, False, True, False, True]),
            "num_mix_category": pd.Categorical([1, "a", True, 1.5, "b"]),
            "num_array_of_numbers": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
            "num_emb1": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb2": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb3": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb4": [1.2, 3.4, 5.6, 7.8, 9.0],
            "target": [7, 15, 8, 25, 99],
        }
    )

    scoring, metric_name, multiplier = _get_scorer_by_name("rmse")

    wrapper = CatBoostWrapper(
        CatBoostRegressor(verbose=False),
        scorer=scoring,
        cat_features=[
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
        ],
        text_features=["text_feature"],
        metric_name=metric_name,
        multiplier=multiplier,
        cv=TimeSeriesSplit(n_splits=2),
        target_type=ModelTaskType.REGRESSION,
    )

    prepared_x, _, _, params = wrapper._prepare_to_fit(df.drop("target", axis=1), df["target"])
    assert set(prepared_x.columns.to_list()) == set(
        [
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "num_float",
            "num_int",
            "num_bool",
            "num_float_category",
            "num_int_category",
            "num_bool_category",
            "num_array_of_numbers",
            "__grouped_embeddings",
        ]
    )
    assert prepared_x["text_feature"].dtype == "object"
    assert prepared_x["cat_float"].dtype == "object"
    assert prepared_x["cat_float_category"].dtype == "object"
    assert prepared_x["cat_int"].dtype == "int64"
    assert prepared_x["cat_int_category"].dtype == "Int64"
    assert prepared_x["cat_str_category"].dtype == "category"
    assert prepared_x["cat_str_object"].dtype == "object"
    assert prepared_x["cat_str"].dtype == "object"
    assert prepared_x["cat_bool"].dtype == "Int64"
    assert prepared_x["cat_bool_category"].dtype == "Int64"
    assert prepared_x["cat_mix"].dtype == "object"
    assert prepared_x["cat_date"].dtype == "object"
    assert prepared_x["num_float"].dtype == "float64"
    assert prepared_x["num_int"].dtype == "int64"
    assert prepared_x["num_bool"].dtype == "Int64"
    assert prepared_x["num_float_category"].dtype == "float64"
    assert prepared_x["num_int_category"].dtype == "int64"
    assert prepared_x["num_bool_category"].dtype == "Int64"
    assert prepared_x["num_array_of_numbers"].dtype == "object"
    assert prepared_x["__grouped_embeddings"].dtype == "object"
    assert set(wrapper.cat_features) == set(
        [
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "cat_float",
            "cat_float_category",
            "cat_int",
        ]
    )
    assert set(wrapper.grouped_embedding_features) == set(["__grouped_embeddings", "num_array_of_numbers"])
    assert set(params["embedding_features"]) == set(["__grouped_embeddings", "num_array_of_numbers"])
    assert params["text_features"] == ["text_feature"]
    assert set(params["cat_features"]) == set(
        [
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "cat_float",
            "cat_float_category",
            "cat_int",
        ]
    )

    prepared_x2, _, params2 = wrapper._prepare_to_calculate(df.drop("target", axis=1), df["target"])
    assert_frame_equal(prepared_x, prepared_x2)
    assert params == params2

    result = wrapper.cross_val_predict(df.drop("target", axis=1), df["target"])
    assert result.get_display_metric() == "51.582 ± 36.412"


def test_lightgbm_wrapper_with_different_feature_types():
    df = pd.DataFrame(
        {
            "constant_feature": [[1], [1], [1], [1], [1]],
            "text_feature": ["a", "b", "c", "a", "b"],
            "cat_float": [1.5, 2.3, 3.1, 4.7, 5.2],
            "cat_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "cat_int": [1, 2, 3, 1, 2],
            "cat_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "cat_str_category": pd.Categorical(["a", "b", "c", "a", "b"]),
            "cat_str_object": ["a", "b", "c", "a", "b"],
            "cat_str": pd.Series(["a", "b", "c", "a", "b"], dtype="string"),
            "cat_bool": [True, False, True, False, True],
            "cat_bool_category": pd.Categorical([True, False, True, False, True]),
            "cat_mix": [1, "a", True, 1.5, "b"],
            "cat_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
            "num_float": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_int": [1, 2, 3, 1, 2],
            "num_str": pd.Series(["x", "y", "z", "x", "y"], dtype="string"),
            "num_str_object": ["x", "y", "z", "x", "y"],
            "num_bool": [True, False, True, False, True],
            "num_mix": [1, "a", True, 1.5, "b"],
            "num_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "num_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "num_str_category": pd.Categorical(["x", "y", "z", "x", "y"]),
            "num_bool_category": pd.Categorical([True, False, True, False, True]),
            "num_mix_category": pd.Categorical([1, "a", True, 1.5, "b"]),
            "num_array_of_numbers": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
            "num_emb1": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb2": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb3": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb4": [1.2, 3.4, 5.6, 7.8, 9.0],
            "target": [0, 1, 0, 1, 0],
        }
    )

    scoring, metric_name, multiplier = _get_scorer_by_name("roc_auc")

    wrapper = LightGBMWrapper(
        LGBMClassifier(verbose=-1),
        scorer=scoring,
        cat_features=[
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
        ],
        text_features=["text_feature"],
        metric_name=metric_name,
        multiplier=multiplier,
        cv=KFold(n_splits=2),
        target_type=ModelTaskType.BINARY,
    )

    prepared_x, _, _, _ = wrapper._prepare_to_fit(df.drop("target", axis=1), df["target"])
    assert set(prepared_x.columns.to_list()) == set(
        [
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "num_float",
            "num_int",
            "num_bool",
            "num_float_category",
            "num_int_category",
            "num_bool_category",
            "num_emb1",
            "num_emb2",
            "num_emb3",
            "num_emb4",
        ]
    )
    assert prepared_x["text_feature"].dtype == "category"
    assert prepared_x["cat_float"].dtype == "category"
    assert prepared_x["cat_float_category"].dtype == "category"
    assert prepared_x["cat_int"].dtype == "category"
    assert prepared_x["cat_int_category"].dtype == "Int64"
    assert prepared_x["cat_str_category"].dtype == "category"
    assert prepared_x["cat_str_object"].dtype == "category"
    assert prepared_x["cat_str"].dtype == "category"
    assert prepared_x["cat_bool"].dtype == "category"
    assert prepared_x["cat_bool_category"].dtype == "category"
    assert prepared_x["cat_mix"].dtype == "category"
    assert prepared_x["cat_date"].dtype == "category"
    assert prepared_x["num_float"].dtype == "float64"
    assert prepared_x["num_int"].dtype == "int64"
    assert prepared_x["num_bool"].dtype == "Int64"
    assert prepared_x["num_float_category"].dtype == "float64"
    assert prepared_x["num_int_category"].dtype == "int64"
    assert prepared_x["num_bool_category"].dtype == "Int64"
    assert prepared_x["num_emb1"].dtype == "float64"
    assert prepared_x["num_emb2"].dtype == "float64"
    assert prepared_x["num_emb3"].dtype == "float64"
    assert prepared_x["num_emb4"].dtype == "float64"
    assert set(wrapper.cat_features) == set(
        [
            "text_feature",
            "cat_str_category",
            "cat_str_object",
            "cat_bool",
            "cat_bool_category",
            "cat_str",
            "cat_mix",
            "cat_date",
            "cat_float",
            "cat_float_category",
            "cat_int",
        ]
    )

    prepared_x2, _, _ = wrapper._prepare_to_calculate(df.drop("target", axis=1), df["target"])
    assert_frame_equal(prepared_x, prepared_x2)

    result = wrapper.cross_val_predict(df.drop("target", axis=1), df["target"])
    assert result.get_display_metric() == "0.500 ± 0.000"


def test_lightgbm_wrapper_with_different_feature_types_time_series():
    df = pd.DataFrame(
        {
            "constant_feature": [[1], [1], [1], [1], [1]],
            "text_feature": ["a", "b", "c", "a", "b"],
            "cat_float": [1.5, 2.3, 3.1, 4.7, 5.2],
            "cat_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "cat_int": [1, 2, 3, 1, 2],
            "cat_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "cat_str_category": pd.Categorical(["a", "b", "c", "a", "b"]),
            "cat_str_object": ["a", "b", "c", "a", "b"],
            "cat_str": pd.Series(["a", "b", "c", "a", "b"], dtype="string"),
            "cat_bool": [True, False, True, False, True],
            "cat_bool_category": pd.Categorical([True, False, True, False, True]),
            "cat_mix": [1, "a", True, 1.5, "b"],
            "cat_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
            "num_float": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_int": [1, 2, 3, 1, 2],
            "num_str": pd.Series(["x", "y", "z", "x", "y"], dtype="string"),
            "num_str_object": ["x", "y", "z", "x", "y"],
            "num_bool": [True, False, True, False, True],
            "num_mix": [1, "a", True, 1.5, "b"],
            "num_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "num_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "num_str_category": pd.Categorical(["x", "y", "z", "x", "y"]),
            "num_bool_category": pd.Categorical([True, False, True, False, True]),
            "num_mix_category": pd.Categorical([1, "a", True, 1.5, "b"]),
            "num_array_of_numbers": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
            "num_emb1": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb2": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb3": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb4": [1.2, 3.4, 5.6, 7.8, 9.0],
            "target": [3, 29, 15, 7, 100],
        }
    )

    scoring, metric_name, multiplier = _get_scorer_by_name("rmse")

    wrapper = LightGBMWrapper(
        LGBMRegressor(verbose=-1),
        scorer=scoring,
        cat_features=[
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
        ],
        text_features=["text_feature"],
        metric_name=metric_name,
        multiplier=multiplier,
        cv=TimeSeriesSplit(n_splits=2),
        target_type=ModelTaskType.REGRESSION,
    )

    prepared_x, _, _, _ = wrapper._prepare_to_fit(df.drop("target", axis=1), df["target"])
    assert set(prepared_x.columns.to_list()) == set(
        [
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "num_float",
            "num_int",
            "num_bool",
            "num_float_category",
            "num_int_category",
            "num_bool_category",
            "num_emb1",
            "num_emb2",
            "num_emb3",
            "num_emb4",
        ]
    )
    assert prepared_x["text_feature"].dtype == "category"
    assert prepared_x["cat_float"].dtype == "category"
    assert prepared_x["cat_float_category"].dtype == "category"
    assert prepared_x["cat_int"].dtype == "category"
    assert prepared_x["cat_int_category"].dtype == "Int64"
    assert prepared_x["cat_str_category"].dtype == "category"
    assert prepared_x["cat_str_object"].dtype == "category"
    assert prepared_x["cat_str"].dtype == "category"
    assert prepared_x["cat_bool"].dtype == "category"
    assert prepared_x["cat_bool_category"].dtype == "category"
    assert prepared_x["cat_mix"].dtype == "category"
    assert prepared_x["cat_date"].dtype == "category"
    assert prepared_x["num_float"].dtype == "float64"
    assert prepared_x["num_int"].dtype == "int64"
    assert prepared_x["num_bool"].dtype == "Int64"
    assert prepared_x["num_float_category"].dtype == "float64"
    assert prepared_x["num_int_category"].dtype == "int64"
    assert prepared_x["num_bool_category"].dtype == "Int64"
    assert prepared_x["num_emb1"].dtype == "float64"
    assert prepared_x["num_emb2"].dtype == "float64"
    assert prepared_x["num_emb3"].dtype == "float64"
    assert prepared_x["num_emb4"].dtype == "float64"
    assert set(wrapper.cat_features) == set(
        [
            "text_feature",
            "cat_str_category",
            "cat_str_object",
            "cat_bool",
            "cat_bool_category",
            "cat_str",
            "cat_mix",
            "cat_date",
            "cat_float",
            "cat_float_category",
            "cat_int",
        ]
    )

    prepared_x2, _, _ = wrapper._prepare_to_calculate(df.drop("target", axis=1), df["target"])
    assert_frame_equal(prepared_x, prepared_x2)

    result = wrapper.cross_val_predict(df.drop("target", axis=1), df["target"])
    assert result.get_display_metric() == "47.583 ± 38.917"


def test_other_wrapper_with_different_feature_types():
    df = pd.DataFrame(
        {
            "constant_feature": [[1], [1], [1], [1], [1]],
            "text_feature": ["a", "b", "c", "a", "b"],
            "cat_float": [1.5, 2.3, 3.1, 4.7, 5.2],
            "cat_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "cat_int": [1, 2, 3, 1, 2],
            "cat_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "cat_str_category": pd.Categorical(["a", "b", "c", "a", "b"]),
            "cat_str_object": ["a", "b", "c", "a", "b"],
            "cat_str": pd.Series(["a", "b", "c", "a", "b"], dtype="string"),
            "cat_bool": [True, False, True, False, True],
            "cat_bool_category": pd.Categorical([True, False, True, False, True]),
            "cat_mix": [1, "a", True, 1.5, "b"],
            "cat_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
            "num_float": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_int": [1, 2, 3, 1, 2],
            "num_str": pd.Series(["x", "y", "z", "x", "y"], dtype="string"),
            "num_str_object": ["x", "y", "z", "x", "y"],
            "num_bool": [True, False, True, False, True],
            "num_mix": [1, "a", True, 1.5, "b"],
            "num_float_category": pd.Categorical([1.5, 2.3, 3.1, 4.7, 5.2]),
            "num_int_category": pd.Categorical([1, 2, 3, 1, 2]),
            "num_str_category": pd.Categorical(["x", "y", "z", "x", "y"]),
            "num_bool_category": pd.Categorical([True, False, True, False, True]),
            "num_mix_category": pd.Categorical([1, "a", True, 1.5, "b"]),
            "num_array_of_numbers": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
            "num_emb1": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb2": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb3": [1.2, 3.4, 5.6, 7.8, 9.0],
            "num_emb4": [1.2, 3.4, 5.6, 7.8, 9.0],
            "target": [0, 1, 0, 1, 0],
        }
    )

    scoring, metric_name, multiplier = _get_scorer_by_name("roc_auc")

    wrapper = OtherEstimatorWrapper(
        RandomForestClassifier(random_state=42),
        scorer=scoring,
        cat_features=[
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
        ],
        text_features=["text_feature"],
        metric_name=metric_name,
        multiplier=multiplier,
        cv=KFold(n_splits=2),
        target_type=ModelTaskType.BINARY,
    )

    prepared_x, _, _, _ = wrapper._prepare_to_fit(df.drop("target", axis=1), df["target"])
    assert set(prepared_x.columns.to_list()) == set(
        [
            "text_feature",
            "cat_float",
            "cat_float_category",
            "cat_int",
            "cat_int_category",
            "cat_str_category",
            "cat_str_object",
            "cat_str",
            "cat_bool",
            "cat_bool_category",
            "cat_mix",
            "cat_date",
            "num_float",
            "num_int",
            "num_bool",
            "num_float_category",
            "num_int_category",
            "num_bool_category",
            "num_emb1",
            "num_emb2",
            "num_emb3",
            "num_emb4",
        ]
    )
    assert prepared_x["text_feature"].dtype == "category"
    assert prepared_x["cat_float"].dtype == "category"
    assert prepared_x["cat_float_category"].dtype == "category"
    assert prepared_x["cat_int"].dtype == "category"
    assert prepared_x["cat_int_category"].dtype == "Int64"
    assert prepared_x["cat_str_category"].dtype == "category"
    assert prepared_x["cat_str_object"].dtype == "category"
    assert prepared_x["cat_str"].dtype == "category"
    assert prepared_x["cat_bool"].dtype == "category"
    assert prepared_x["cat_bool_category"].dtype == "category"
    assert prepared_x["cat_mix"].dtype == "category"
    assert prepared_x["cat_date"].dtype == "category"
    assert prepared_x["num_float"].dtype == "float64"
    assert prepared_x["num_int"].dtype == "int64"
    assert prepared_x["num_bool"].dtype == "Int64"
    assert prepared_x["num_float_category"].dtype == "float64"
    assert prepared_x["num_int_category"].dtype == "int64"
    assert prepared_x["num_bool_category"].dtype == "Int64"
    assert prepared_x["num_emb1"].dtype == "float64"
    assert prepared_x["num_emb2"].dtype == "float64"
    assert prepared_x["num_emb3"].dtype == "float64"
    assert prepared_x["num_emb4"].dtype == "float64"
    assert set(wrapper.cat_features) == set(
        [
            "text_feature",
            "cat_str_category",
            "cat_str_object",
            "cat_bool",
            "cat_bool_category",
            "cat_str",
            "cat_mix",
            "cat_date",
            "cat_float",
            "cat_float_category",
            "cat_int",
        ]
    )

    prepared_x2, _, _ = wrapper._prepare_to_calculate(df.drop("target", axis=1), df["target"])
    assert_frame_equal(prepared_x, prepared_x2)

    result = wrapper.cross_val_predict(df.drop("target", axis=1), df["target"])
    assert result.get_display_metric() == "0.750 ± 0.250"
