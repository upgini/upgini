import os

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from pandas.testing import assert_frame_equal
from requests_mock.mocker import Mocker
from sklearn.ensemble import RandomForestClassifier

from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType

from .utils import (
    mock_default_requests,
    mock_get_features_meta,
    mock_get_metadata,
    mock_initial_search,
    mock_initial_summary,
    mock_raw_features,
    mock_validation_raw_features,
    mock_validation_search,
    mock_validation_summary,
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_data/enricher/",
)


def test_real_case_metric_binary(requests_mock: Mocker):
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data/",
    )

    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=100.0,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0},
        ],
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
                    "meaningType": "DATE"
                },
                {
                    "index": 2,
                    "name": "target",
                    "originalName": "target",
                    "dataType": "INT",
                    "meaningType": "DATE"
                },
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
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[],
        etalon_features=[{"name": "score", "importance": 0.368092, "matchedInPercent": 100.0, "valueType": "NUMERIC"}],
    )
    # path_to_mock_features = os.path.join(BASE_DIR, "features.parquet")
    # mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    train = pd.read_parquet(os.path.join(BASE_DIR, "real_train.parquet"))
    X = train[["request_date", "score"]]
    y = train["target1"]
    test = pd.read_parquet(os.path.join(BASE_DIR, "real_test.parquet"))
    eval_set = [(test[["request_date", "score"]], test["target1"])]

    enricher = FeaturesEnricher(
        search_keys={"request_date": SearchKey.DATE},
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
        country_code="RU",
        search_id=search_task_id,
        logs_enabled=False
    )

    enriched_X = pd.read_parquet(os.path.join(BASE_DIR, "real_enriched_x.parquet"))
    enricher.enriched_X = enriched_X

    enriched_eval_x = pd.read_parquet(os.path.join(BASE_DIR, "real_enriched_eval_x.parquet"))
    enricher.enriched_eval_sets = {1: enriched_eval_x}

    metrics = enricher.calculate_metrics(X, y, eval_set)
    print(metrics)

    expected_metrics = pd.DataFrame({
        "segment": ["train", "eval 1"],
        "match_rate": [100.0, 100.0],
        "baseline roc_auc": [0.743380, 0.721769]
    }).set_index("segment").rename_axis("")

    assert_frame_equal(expected_metrics, metrics)


def test_default_metric_binary(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.0,
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
        ads_features=[{"name": "ads_feature1", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[{"name": "feature1", "importance": 0.1, "matchedInPercent": 100.0, "valueType": "NUMERIC"}],
    )
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.0,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
    mock_validation_raw_features(requests_mock, url, validation_search_task_id, path_to_mock_validation_features)

    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input.csv"))
    df["feature_2_cat"] = np.random.randint(0, 10, len(df))
    df["feature_2_cat"] = df["feature_2_cat"].astype(str).astype("category")
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
        logs_enabled=False
    )

    # with pytest.raises(Exception, match="Fit the enricher before calling calculate_metrics."):
    # enricher.calculate_metrics(X, y)

    enriched_X = enricher.fit_transform(X, y, eval_set)

    assert len(enriched_X) == len(X)

    assert len(enricher.enriched_eval_sets) == 2
    assert len(enricher.enriched_eval_sets[1]) == 250
    assert len(enricher.enriched_eval_sets[2]) == 250

    metrics_df = enricher.calculate_metrics(X, y, eval_set)
    assert metrics_df is not None
    print(metrics_df)
    assert metrics_df.loc["train", "match_rate"] == 99.0

    assert metrics_df.loc["train", "baseline roc_auc"] == approx(0.49976)
    assert metrics_df.loc["train", "enriched roc_auc"] == approx(0.503121)
    assert metrics_df.loc["train", "uplift"] == approx(0.003361)

    assert metrics_df.loc["eval 1", "match_rate"] == 100.0
    assert metrics_df.loc["eval 1", "baseline roc_auc"] == approx(0.5)
    assert metrics_df.loc["eval 1", "enriched roc_auc"] == approx(0.5)
    assert metrics_df.loc["eval 1", "uplift"] == approx(0.0)

    assert metrics_df.loc["eval 2", "match_rate"] == 99.0
    assert metrics_df.loc["eval 2", "baseline roc_auc"] == approx(0.5)
    assert metrics_df.loc["eval 2", "enriched roc_auc"] == approx(0.5)
    assert metrics_df.loc["eval 2", "uplift"] == approx(0.0)


def test_blocked_timeseries_rmsle(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.0,
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
        ads_features=[{"name": "ads_feature1", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[{"name": "feature1", "importance": 0.1, "matchedInPercent": 100.0, "valueType": "NUMERIC"}],
    )
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.0,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
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
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        cv=CVType.blocked_time_series,
        logs_enabled=False,
    )

    enriched_X = enricher.fit_transform(X, y, eval_set)

    assert len(enriched_X) == len(X)

    assert len(enricher.enriched_eval_sets) == 2
    assert len(enricher.enriched_eval_sets[1]) == 250
    assert len(enricher.enriched_eval_sets[2]) == 250

    metrics_df = enricher.calculate_metrics(X, y, eval_set, scoring="RMSLE")
    assert metrics_df is not None
    print(metrics_df)
    assert metrics_df.loc["train", "match_rate"] == 99.0
    assert metrics_df.loc["train", "baseline RMSLE"] == approx(0.487154)
    assert metrics_df.loc["train", "enriched RMSLE"] == approx(0.478443)
    assert metrics_df.loc["train", "uplift"] == approx(0.008710)

    assert metrics_df.loc["eval 1", "match_rate"] == 100.0
    assert metrics_df.loc["eval 1", "baseline RMSLE"] == approx(0.475431)
    assert metrics_df.loc["eval 1", "enriched RMSLE"] == approx(0.484852)
    assert metrics_df.loc["eval 1", "uplift"] == approx(-0.009421)

    assert metrics_df.loc["eval 2", "match_rate"] == 99.0
    assert metrics_df.loc["eval 2", "baseline RMSLE"] == approx(0.500405)
    assert metrics_df.loc["eval 2", "enriched RMSLE"] == approx(0.493342)
    assert metrics_df.loc["eval 2", "uplift"] == approx(0.007062)


def test_catboost_metric_binary(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.0,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.5},
        ],
    )
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "ads_feature1", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[{"name": "feature1", "importance": 0.1, "matchedInPercent": 100.0, "valueType": "NUMERIC"}],
    )
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.0,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
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

    with pytest.raises(Exception, match="Fit the enricher before calling calculate_metrics."):
        enricher.calculate_metrics(X, y)

    enriched_X = enricher.fit_transform(X, y, eval_set)

    assert len(enriched_X) == len(X)

    assert len(enricher.enriched_eval_sets) == 2
    assert len(enricher.enriched_eval_sets[1]) == 250
    assert len(enricher.enriched_eval_sets[2]) == 250

    estimator = CatBoostClassifier(random_seed=42, verbose=False)
    metrics_df = enricher.calculate_metrics(X, y, eval_set, estimator=estimator, scoring="roc_auc")
    assert metrics_df is not None
    print(metrics_df)

    assert metrics_df.loc["train", "match_rate"] == 99.0
    assert metrics_df.loc["train", "baseline roc_auc"] == approx(0.497839)
    assert metrics_df.loc["train", "enriched roc_auc"] == approx(0.490196)
    assert metrics_df.loc["train", "uplift"] == approx(-0.007643)

    assert metrics_df.loc["eval 1", "match_rate"] == 100.0
    assert metrics_df.loc["eval 1", "baseline roc_auc"] == approx(0.500000)
    assert metrics_df.loc["eval 1", "enriched roc_auc"] == approx(0.500000)
    assert metrics_df.loc["eval 1", "uplift"] == approx(0.0)

    assert metrics_df.loc["eval 2", "match_rate"] == 99.0
    assert metrics_df.loc["eval 2", "baseline roc_auc"] == approx(0.500000)
    assert metrics_df.loc["eval 2", "enriched roc_auc"] == approx(0.500000)
    assert metrics_df.loc["eval 2", "uplift"] == approx(0.0)


@pytest.mark.skip()
def test_lightgbm_metric_binary(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.0,
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
        ads_features=[{"name": "ads_feature1", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[{"name": "feature1", "importance": 0.1, "matchedInPercent": 100.0, "valueType": "NUMERIC"}],
    )
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.0,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
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

    with pytest.raises(Exception, match="Fit wasn't completed successfully"):
        enricher.calculate_metrics(X, y)

    enriched_X = enricher.fit_transform(X, y, eval_set)

    assert len(enriched_X) == len(X)

    assert len(enricher.enriched_eval_sets) == 2
    assert len(enricher.enriched_eval_sets[1]) == 250
    assert len(enricher.enriched_eval_sets[2]) == 250

    from lightgbm import LGBMClassifier  # type: ignore

    estimator = LGBMClassifier(random_seed=42)
    metrics_df = enricher.calculate_metrics(X, y, eval_set, estimator=estimator, scoring="mean_absolute_error")
    assert metrics_df is not None
    print(metrics_df)
    assert metrics_df.loc["train", "match_rate"] == 99.0
    assert metrics_df.loc["train", "baseline roc_auc"] == approx(0.476230)  # Investigate same values
    assert metrics_df.loc["train", "enriched roc_auc"] == approx(0.476230)
    assert metrics_df.loc["train", "uplift"] == approx(0.0)

    assert metrics_df.loc["eval 1", "match_rate"] == 100.0
    assert metrics_df.loc["eval 1", "baseline roc_auc"] == approx(0.500872)
    assert metrics_df.loc["eval 1", "enriched roc_auc"] == approx(0.500872)
    assert metrics_df.loc["eval 1", "uplift"] == approx(0.0)

    assert metrics_df.loc["eval 2", "match_rate"] == 99.0
    assert metrics_df.loc["eval 2", "baseline roc_auc"] == approx(0.521455)
    assert metrics_df.loc["eval 2", "enriched roc_auc"] == approx(0.521455)
    assert metrics_df.loc["eval 2", "uplift"] == approx(0.0)


def test_rf_metric_rmse(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(
        requests_mock,
        url,
        search_task_id,
        hit_rate=99.0,
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
        ads_features=[{"name": "ads_feature1", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[{"name": "feature1", "importance": 0.1, "matchedInPercent": 100.0, "valueType": "NUMERIC"}],
    )
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    validation_search_task_id = mock_validation_search(requests_mock, url, search_task_id)
    mock_validation_summary(
        requests_mock,
        url,
        search_task_id,
        ads_search_task_id,
        validation_search_task_id,
        hit_rate=99.0,
        auc=0.66,
        uplift=0.1,
        eval_set_metrics=[
            {"eval_set_index": 1, "hit_rate": 1.0, "auc": 0.5},
            {"eval_set_index": 2, "hit_rate": 0.99, "auc": 0.77},
        ],
    )
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
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

    with pytest.raises(Exception, match="Fit the enricher before calling calculate_metrics."):
        enricher.calculate_metrics(X, y)

    enriched_X = enricher.fit_transform(X, y, eval_set)

    assert len(enriched_X) == len(X)

    assert len(enricher.enriched_eval_sets) == 2
    assert len(enricher.enriched_eval_sets[1]) == 250
    assert len(enricher.enriched_eval_sets[2]) == 250

    estimator = RandomForestClassifier(random_state=42)
    metrics_df = enricher.calculate_metrics(X, y, eval_set, estimator=estimator, scoring="rmse")
    assert metrics_df is not None
    print(metrics_df)
    baseline_metric = "baseline rmse"
    enriched_metric = "enriched rmse"
    assert metrics_df.loc["train", "match_rate"] == 99.0
    assert metrics_df.loc["train", baseline_metric] == approx(0.706528)
    assert metrics_df.loc["train", enriched_metric] == approx(0.702535)
    assert metrics_df.loc["train", "uplift"] == approx(0.003993)

    assert metrics_df.loc["eval 1", "match_rate"] == 100.0
    assert metrics_df.loc["eval 1", baseline_metric] == approx(0.672309)
    assert metrics_df.loc["eval 1", enriched_metric] == approx(0.672309)
    assert metrics_df.loc["eval 1", "uplift"] == approx(0.0)

    assert metrics_df.loc["eval 2", "match_rate"] == 99.0
    assert metrics_df.loc["eval 2", baseline_metric] == approx(0.732120)
    assert metrics_df.loc["eval 2", enriched_metric] == approx(0.732120)
    assert metrics_df.loc["eval 2", "uplift"] == approx(0.0)


def approx(value: float):
    return pytest.approx(value, abs=0.000001)
