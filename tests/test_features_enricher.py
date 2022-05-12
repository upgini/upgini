import os

import pandas as pd
import pytest
from requests_mock.mocker import Mocker

from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import RuntimeParameters

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


def test_search_keys_validation(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    with pytest.raises(Exception, match="Date and datetime search keys are presented simultaniously"):
        FeaturesEnricher(
            search_keys={"d1": SearchKey.DATE, "dt2": SearchKey.DATETIME},
            endpoint=url,
        )

    with pytest.raises(Exception, match="COUNTRY search key should be provided if POSTAL_CODE is presented"):
        FeaturesEnricher(search_keys={"postal_code": SearchKey.POSTAL_CODE}, endpoint=url)


def test_features_enricher(requests_mock: Mocker):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.csv.gz"
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
        etalon_features=[{"name": "SystemRecordId_473310000", "importance": 1.0, "matchedInPercent": 100.0}],
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
        keep_input=True,
        endpoint=url,
        api_key="fake_api_key",
        date_format="%Y-%m-%d",
    )

    enriched_train_features = enricher.fit_transform(
        train_features, train_target, eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)]
    )
    assert enriched_train_features.shape == (10000, 4)

    metrics = enricher.calculate_metrics(
        train_features, train_target, eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)]
    )
    expected_metrics = pd.DataFrame(
        [
            {
                "match_rate": 99.9,
                "baseline roc_auc": 0.5,
                "enriched roc_auc": 0.4926257640349131,
                "uplift": -0.007374235965086906,
            },
            {"match_rate": 100.0, "baseline roc_auc": 0.5, "enriched roc_auc": 0.5, "uplift": 0.0},
            {"match_rate": 99.0, "baseline roc_auc": 0.5, "enriched roc_auc": 0.5, "uplift": 0.0},
        ],
        index=["train", "eval 1", "eval 2"],
    )
    print("Expected metrics: ")
    print(expected_metrics)
    print("Actual metrics: ")
    print(metrics)

    assert metrics is not None
    for segment in expected_metrics.index:
        for col in expected_metrics.columns:
            assert metrics.loc[segment, col] == expected_metrics.loc[segment, col]

    print(enricher.features_info)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 2
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info["feature_name"] == "feature"
    assert first_feature_info["shap_value"] == 10.1
    second_feature_info = enricher.features_info.iloc[1]
    assert second_feature_info["feature_name"] == "SystemRecordId_473310000"
    assert second_feature_info["shap_value"] == 1.0


def test_features_enricher_fit_transform_runtime_parameters(requests_mock: Mocker):
    url = "http://fake_url2"
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.csv.gz"
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
        keep_input=True,
        date_format="%Y-%m-%d",
        endpoint=url,
        api_key="fake_api_key",
        runtime_parameters=RuntimeParameters(properties={"runtimeProperty1": "runtimeValue1"}),
    )
    assert enricher.runtime_parameters is not None

    enricher.fit(
        train_features, train_target, eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)]
    )

    fit_req = None
    initial_search_url = url + "/public/api/v2/search/initial"
    for elem in requests_mock.request_history:
        if elem.url == initial_search_url:
            fit_req = elem

    # TODO: can be better with
    #  https://metareal.blog/en/post/2020/05/03/validating-multipart-form-data-with-requests-mock/
    # It's do-able to parse req with cgi module and verify contents
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

    transformed = enricher.transform(train_features)

    transform_req = None
    transform_url = url + "/public/api/v2/search/validation?initialSearchTaskId=" + search_task_id
    for elem in requests_mock.request_history:
        if elem.url == transform_url:
            transform_req = elem

    assert transform_req is not None
    assert "runtimeProperty1" in str(transform_req.body)
    assert "runtimeValue1" in str(transform_req.body)

    assert transformed.shape == (10000, 4)
