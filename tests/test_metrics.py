import datetime
import os

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from pandas.testing import assert_frame_equal
from requests_mock.mocker import Mocker
from sklearn.ensemble import RandomForestClassifier

from upgini import FeaturesEnricher, SearchKey
from upgini.errors import ValidationError
from upgini.metadata import (
    CVType,
    FeaturesMetadataV2,
    HitRateMetrics,
    ModelEvalSet,
    ProviderTaskMetadataV2,
)
from upgini.resource_bundle import bundle
from upgini.search_task import SearchTask

from .utils import (
    mock_default_requests,
    mock_get_metadata,
    mock_get_task_metadata_v2,
    mock_initial_progress,
    mock_initial_search,
    mock_initial_summary,
    mock_raw_features,
    mock_validation_progress,
    mock_validation_raw_features,
    mock_validation_search,
    mock_validation_summary,
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_data/enricher/",
)

segment_header = bundle.get("quality_metrics_segment_header")
train_segment = bundle.get("quality_metrics_train_segment")
eval_1_segment = bundle.get("quality_metrics_eval_segment").format(1)
eval_2_segment = bundle.get("quality_metrics_eval_segment").format(2)
rows_header = bundle.get("quality_metrics_rows_header")
target_mean_header = bundle.get("quality_metrics_mean_target_header")
match_rate_header = bundle.get("quality_metrics_match_rate_header")
baseline_rocauc = bundle.get("quality_metrics_baseline_header").format("roc_auc")
enriched_rocauc = bundle.get("quality_metrics_enriched_header").format("roc_auc")
baseline_gini = bundle.get("quality_metrics_baseline_header").format("GINI")
enriched_gini = bundle.get("quality_metrics_enriched_header").format("GINI")
baseline_rmse = bundle.get("quality_metrics_baseline_header").format("rmse")
enriched_rmse = bundle.get("quality_metrics_enriched_header").format("rmse")
baseline_RMSLE = bundle.get("quality_metrics_baseline_header").format("RMSLE")
enriched_RMSLE = bundle.get("quality_metrics_enriched_header").format("RMSLE")
baseline_mae = bundle.get("quality_metrics_baseline_header").format("mean_absolute_error")
enriched_mae = bundle.get("quality_metrics_enriched_header").format("mean_absolute_error")
uplift = bundle.get("quality_metrics_uplift_header")

SearchTask.PROTECT_FROM_RATE_LIMIT = False
SearchTask.POLLING_DELAY_SECONDS = 0.1


def test_real_case_metric_binary(requests_mock: Mocker):
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
    path_to_mock_features = os.path.join(BASE_DIR, "real_enriched_x.parquet")
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    train = pd.read_parquet(os.path.join(BASE_DIR, "real_train.parquet"))
    X = train[["request_date", "score"]]
    y = train["target1"].rename("target")
    test = pd.read_parquet(os.path.join(BASE_DIR, "real_test.parquet"))
    eval_set = [(test[["request_date", "score"]], test["target1"].rename("target"))]

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

    enricher.X = X
    enricher.y = y
    enricher.eval_set = eval_set

    enriched_X = pd.read_parquet(os.path.join(BASE_DIR, "real_enriched_x.parquet"))
    enriched_eval_x = pd.read_parquet(os.path.join(BASE_DIR, "real_enriched_eval_x.parquet"))

    sampled_Xy = X.copy()
    sampled_Xy["target"] = y
    sampled_Xy = sampled_Xy[sampled_Xy.index.isin(enriched_X.index)]
    sampled_X = sampled_Xy.drop(columns="target")
    sampled_y = sampled_Xy["target"]
    enricher._FeaturesEnricher__cached_sampled_datasets = (
        sampled_X,
        sampled_y,
        enriched_X,
        {0: (eval_set[0][0], enriched_eval_x, eval_set[0][1])},
        search_keys,
    )

    metrics = enricher.calculate_metrics()
    print(metrics)

    expected_metrics = pd.DataFrame(
        {
            segment_header: [train_segment, eval_1_segment],
            rows_header: [6582, 2505],
            target_mean_header: [0.5, 0.8854],
            baseline_gini: [0.486038, 0.450277],
        }
    )

    assert_frame_equal(expected_metrics, metrics)


def test_default_metric_binary(requests_mock: Mocker):
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
                    name="feature1",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
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
    )

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    metrics_df = enricher.calculate_metrics()
    assert metrics_df is not None
    print(metrics_df)

    # FIXME: different between python versions
    # assert metrics_df.loc[0, segment_header] == train_segment
    # assert metrics_df.loc[0, rows_header] == 500
    # assert metrics_df.loc[0, target_mean_header] == 0.51
    # assert metrics_df.loc[0, baseline_gini] == approx(0.104954)
    # assert metrics_df.loc[0, enriched_gini] == approx(0.097089)
    # assert metrics_df.loc[0, uplift] == approx(-0.007864)

    # assert metrics_df.loc[1, segment_header] == eval_1_segment
    # assert metrics_df.loc[1, rows_header] == 250
    # assert metrics_df.loc[1, target_mean_header] == 0.452
    # assert metrics_df.loc[1, baseline_gini] == approx(-0.053705)
    # assert metrics_df.loc[1, enriched_gini] == approx(0.080266)
    # assert metrics_df.loc[1, uplift] == approx(0.133971)

    # assert metrics_df.loc[2, segment_header] == eval_2_segment
    # assert metrics_df.loc[2, rows_header] == 250
    # assert metrics_df.loc[2, target_mean_header] == 0.536
    # assert metrics_df.loc[2, baseline_gini] == approx(-0.002072)
    # assert metrics_df.loc[2, enriched_gini] == approx(-0.002432)
    # assert metrics_df.loc[2, uplift] == approx(-0.000360)


def test_default_metric_binary_custom_loss(requests_mock: Mocker):
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
                    name="feature1",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
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
    print(metrics_df)

    # FIXME: different between python versions
    # assert metrics_df.loc[0, segment_header] == train_segment
    # assert metrics_df.loc[0, rows_header] == 500
    # assert metrics_df.loc[0, target_mean_header] == 0.51
    # assert metrics_df.loc[0, baseline_gini] == approx(0.104954)
    # assert metrics_df.loc[0, enriched_gini] == approx(0.097089)
    # assert metrics_df.loc[0, uplift] == approx(-0.007864)

    # assert metrics_df.loc[1, segment_header] == eval_1_segment
    # assert metrics_df.loc[1, rows_header] == 250
    # assert metrics_df.loc[1, target_mean_header] == 0.452
    # assert metrics_df.loc[1, baseline_gini] == approx(-0.053705)
    # assert metrics_df.loc[1, enriched_gini] == approx(0.080266)
    # assert metrics_df.loc[1, uplift] == approx(0.133971)

    # assert metrics_df.loc[2, segment_header] == eval_2_segment
    # assert metrics_df.loc[2, rows_header] == 250
    # assert metrics_df.loc[2, target_mean_header] == 0.536
    # assert metrics_df.loc[2, baseline_gini] == approx(-0.002072)
    # assert metrics_df.loc[2, enriched_gini] == approx(-0.002432)
    # assert metrics_df.loc[2, uplift] == approx(-0.000360)


def test_default_metric_binary_shuffled(requests_mock: Mocker):
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
                    name="feature1",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.0
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
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
    print(metrics_df)

    # FIXME: different between python versions
    # assert metrics_df.loc[0, segment_header] == train_segment
    # assert metrics_df.loc[0, rows_header] == 500
    # assert metrics_df.loc[0, target_mean_header] == 0.51
    # assert metrics_df.loc[0, baseline_gini] == approx(0.104954)
    # assert metrics_df.loc[0, enriched_gini] == approx(0.097089)
    # assert metrics_df.loc[0, uplift] == approx(-0.007864)

    # assert metrics_df.loc[1, segment_header] == eval_1_segment
    # assert metrics_df.loc[1, rows_header] == 250
    # assert metrics_df.loc[1, target_mean_header] == 0.452
    # assert metrics_df.loc[1, baseline_gini] == approx(-0.053705)
    # assert metrics_df.loc[1, enriched_gini] == approx(0.080266)
    # assert metrics_df.loc[1, uplift] == approx(0.133971)

    # assert metrics_df.loc[2, segment_header] == eval_2_segment
    # assert metrics_df.loc[2, rows_header] == 250
    # assert metrics_df.loc[2, target_mean_header] == 0.536
    # assert metrics_df.loc[2, baseline_gini] == approx(-0.002072)
    # assert metrics_df.loc[2, enriched_gini] == approx(-0.002432)
    # assert metrics_df.loc[2, uplift] == approx(-0.000360)


def test_blocked_timeseries_rmsle(requests_mock: Mocker):
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
                    name="feature1",
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    metrics_df = enricher.calculate_metrics(scoring="RMSLE")
    assert metrics_df is not None
    print(metrics_df)

    assert metrics_df.loc[0, segment_header] == train_segment
    assert metrics_df.loc[0, rows_header] == 500
    assert metrics_df.loc[0, target_mean_header] == 0.51
    assert metrics_df.loc[0, baseline_RMSLE] == approx(0.487498)
    assert metrics_df.loc[0, enriched_RMSLE] == approx(0.441673)
    assert metrics_df.loc[0, uplift] == approx(0.045825)

    assert metrics_df.loc[1, segment_header] == eval_1_segment
    assert metrics_df.loc[1, rows_header] == 250
    assert metrics_df.loc[1, target_mean_header] == 0.452
    assert metrics_df.loc[1, baseline_RMSLE] == approx(0.499389)
    assert metrics_df.loc[1, enriched_RMSLE] == approx(0.475467)
    assert metrics_df.loc[1, uplift] == approx(0.023922)

    assert metrics_df.loc[2, segment_header] == eval_2_segment
    assert metrics_df.loc[2, rows_header] == 250
    assert metrics_df.loc[2, target_mean_header] == 0.536
    assert metrics_df.loc[2, baseline_RMSLE] == approx(0.497084)
    assert metrics_df.loc[2, enriched_RMSLE] == approx(0.492340)
    assert metrics_df.loc[2, uplift] == approx(0.004744)


def test_catboost_metric_binary(requests_mock: Mocker):
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
                    name="feature1",
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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

    with pytest.raises(ValidationError, match=bundle.get("metrics_unfitted_enricher")):
        enricher.calculate_metrics()

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    estimator = CatBoostClassifier(random_seed=42, verbose=False)
    metrics_df = enricher.calculate_metrics(estimator=estimator, scoring="roc_auc")
    assert metrics_df is not None
    print(metrics_df)

    assert metrics_df.loc[0, segment_header] == train_segment
    assert metrics_df.loc[0, rows_header] == 500
    assert metrics_df.loc[0, target_mean_header] == 0.51
    assert metrics_df.loc[0, baseline_gini] == approx(0.023101)
    assert metrics_df.loc[0, enriched_gini] == approx(0.090344)
    assert metrics_df.loc[0, uplift] == approx(0.067243)

    assert metrics_df.loc[1, segment_header] == eval_1_segment
    assert metrics_df.loc[1, rows_header] == 250
    assert metrics_df.loc[1, target_mean_header] == 0.452
    assert metrics_df.loc[1, baseline_gini] == approx(-0.016188)
    assert metrics_df.loc[1, enriched_gini] == approx(0.014947)
    assert metrics_df.loc[1, uplift] == approx(0.031135)

    assert metrics_df.loc[2, segment_header] == eval_2_segment
    assert metrics_df.loc[2, rows_header] == 250
    assert metrics_df.loc[2, target_mean_header] == 0.536
    assert metrics_df.loc[2, baseline_gini] == approx(-0.017138)
    assert metrics_df.loc[2, enriched_gini] == approx(0.035666)
    assert metrics_df.loc[2, uplift] == approx(0.052805)


def test_catboost_metric_binary_with_cat_features(requests_mock: Mocker):
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
                    name="cat_feature2",
                    type="categorical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.2,
                ),
                FeaturesMetadataV2(
                    name="feature1",
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
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

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    estimator = CatBoostClassifier(random_seed=42, verbose=False, cat_features=[1, 3])
    metrics_df = enricher.calculate_metrics(estimator=estimator, scoring="roc_auc")
    assert metrics_df is not None
    print(metrics_df)

    assert metrics_df.loc[0, segment_header] == train_segment
    assert metrics_df.loc[0, rows_header] == 500
    assert metrics_df.loc[0, target_mean_header] == 0.51
    assert metrics_df.loc[0, baseline_gini] == approx(0.102928)
    assert metrics_df.loc[0, enriched_gini] == approx(0.139437)
    assert metrics_df.loc[0, uplift] == approx(0.036508)

    assert metrics_df.loc[1, segment_header] == eval_1_segment
    assert metrics_df.loc[1, rows_header] == 250
    assert metrics_df.loc[1, target_mean_header] == 0.452
    assert metrics_df.loc[1, baseline_gini] == approx(-0.074491)
    assert metrics_df.loc[1, enriched_gini] == approx(-0.052619)
    assert metrics_df.loc[1, uplift] == approx(0.021872)

    assert metrics_df.loc[2, segment_header] == eval_2_segment
    assert metrics_df.loc[2, rows_header] == 250
    assert metrics_df.loc[2, target_mean_header] == 0.536
    assert metrics_df.loc[2, baseline_gini] == approx(0.022002)
    assert metrics_df.loc[2, enriched_gini] == approx(-0.010950)
    assert metrics_df.loc[2, uplift] == approx(-0.032952)


@pytest.mark.skip()
def test_lightgbm_metric_binary(requests_mock: Mocker):
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
                    name="feature1",
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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
    pd.set_option("display.max_columns", 1000)
    print(metrics_df)

    assert metrics_df.loc[0, segment_header] == "Train"
    assert metrics_df.loc[0, rows_header] == 500
    assert metrics_df.loc[0, target_mean_header] == 0.51
    assert metrics_df.loc[0, baseline_mae] == approx(0.4980)
    assert metrics_df.loc[0, enriched_mae] == approx(0.4960)
    assert metrics_df.loc[0, uplift] == approx(0.002)

    assert metrics_df.loc[1, segment_header] == "Eval 1"
    assert metrics_df.loc[1, rows_header] == 250
    assert metrics_df.loc[1, target_mean_header] == 0.452
    assert metrics_df.loc[1, baseline_mae] == approx(0.4752)
    assert metrics_df.loc[1, enriched_mae] == approx(0.4912)
    assert metrics_df.loc[1, uplift] == approx(-0.016)

    assert metrics_df.loc[2, segment_header] == "Eval 2"
    assert metrics_df.loc[2, rows_header] == 250
    assert metrics_df.loc[2, target_mean_header] == 0.536
    assert metrics_df.loc[2, baseline_mae] == approx(0.5)
    assert metrics_df.loc[2, enriched_mae] == approx(0.4696)
    assert metrics_df.loc[2, uplift] == approx(0.0304)


def test_rf_metric_rmse(requests_mock: Mocker):
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
                    name="feature1",
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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

    with pytest.raises(ValidationError, match=bundle.get("metrics_unfitted_enricher")):
        enricher.calculate_metrics()

    enriched_X = enricher.fit_transform(X, y, eval_set, calculate_metrics=False)

    assert len(enriched_X) == len(X)

    estimator = RandomForestClassifier(random_state=42)
    metrics_df = enricher.calculate_metrics(estimator=estimator, scoring="rmse")
    assert metrics_df is not None
    print(metrics_df)

    assert metrics_df.loc[0, segment_header] == train_segment
    assert metrics_df.loc[0, rows_header] == 500
    assert metrics_df.loc[0, target_mean_header] == 0.51
    assert metrics_df.loc[0, baseline_rmse] == approx(0.737054)
    assert metrics_df.loc[0, enriched_rmse] == approx(0.720624)
    assert metrics_df.loc[0, uplift] == approx(0.016430)

    assert metrics_df.loc[1, segment_header] == eval_1_segment
    assert metrics_df.loc[1, rows_header] == 250
    assert metrics_df.loc[1, target_mean_header] == 0.452
    assert metrics_df.loc[1, baseline_rmse] == approx(0.704719)
    assert metrics_df.loc[1, enriched_rmse] == approx(0.721444)
    assert metrics_df.loc[1, uplift] == approx(-0.016725)

    assert metrics_df.loc[2, segment_header] == eval_2_segment
    assert metrics_df.loc[2, rows_header] == 250
    assert metrics_df.loc[2, target_mean_header] == 0.536
    assert metrics_df.loc[2, baseline_rmse] == approx(0.690261)
    assert metrics_df.loc[2, enriched_rmse] == approx(0.694711)
    assert metrics_df.loc[2, uplift] == approx(-0.004450)


def test_default_metric_binary_with_string_feature(requests_mock: Mocker):
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
                    name="feature1",
                    type="numerical",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.1,
                ),
                FeaturesMetadataV2(
                    name="feature_2_cat", type="categorical", source="etalon", hit_rate=100.0, shap_value=0.01
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
    path_to_mock_features = os.path.join(FIXTURE_DIR, "features.parquet")
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
    path_to_mock_validation_features = os.path.join(FIXTURE_DIR, "validation_features.parquet")
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
    print(metrics_df)

    assert metrics_df.loc[0, segment_header] == train_segment
    assert metrics_df.loc[0, rows_header] == 500
    assert metrics_df.loc[0, target_mean_header] == 0.51
    assert metrics_df.loc[0, baseline_gini] == approx(0.103614)
    assert metrics_df.loc[0, enriched_gini] == approx(0.090759)
    assert metrics_df.loc[0, uplift] == approx(-0.012855)

    assert metrics_df.loc[1, segment_header] == eval_1_segment
    assert metrics_df.loc[1, rows_header] == 250
    assert metrics_df.loc[1, target_mean_header] == 0.452
    assert metrics_df.loc[1, baseline_gini] == approx(-0.109605)
    assert metrics_df.loc[1, enriched_gini] == approx(-0.011860)
    assert metrics_df.loc[1, uplift] == approx(0.097746)

    assert metrics_df.loc[2, segment_header] == eval_2_segment
    assert metrics_df.loc[2, rows_header] == 250
    assert metrics_df.loc[2, target_mean_header] == 0.536
    assert metrics_df.loc[2, baseline_gini] == approx(-0.008801)
    assert metrics_df.loc[2, enriched_gini] == approx(-0.011220)
    assert metrics_df.loc[2, uplift] == approx(-0.002419)


def approx(value: float):
    return pytest.approx(value, abs=0.000001)
