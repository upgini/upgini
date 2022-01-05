import os

import pandas as pd

from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import RuntimeParameters
from requests_mock.mocker import Mocker


def test_features_enricher(requests_mock):
    url = "http://fake_url2"

    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.csv.gz"
    )

    requests_mock.post(url + "/private/api/v2/security/refresh_access_token", json={"access_token": "123"})
    requests_mock.post(
        url + "/public/api/v2/search/initial",
        json={
            "fileUploadId": "123",
            "searchTaskId": "321",
            "searchType": "INITIAL",
            "status": "SUBMITTED",
            "extractFeatures": "true",
            "returnScores": "false",
            "createdAt": 1633302145414,
        },
    )
    requests_mock.get(
        url + "/public/api/v2/search/321",
        json={
            "fileUploadTaskId": "123",
            "searchTaskId": "321",
            "searchTaskStatus": "COMPLETED",
            "featuresFoundCount": 1,
            "providersCheckedCount": 1,
            "importantProvidersCount": 1,
            "importantFeaturesCount": 1,
            "importantProviders": [
                {
                    "adsSearchTaskId": "432",
                    "searchTaskId": "321",
                    "searchType": "INITIAL",
                    "taskStatus": "COMPLETED",
                    "providerName": "Provider-123456",
                    "providerId": "123456",
                    "providerQuality": {
                        "metrics": [
                            {"code": "HIT_RATE", "value": 99.9},
                            {"code": "AUC", "value": 0.66},
                            {"code": "UPLIFT", "value": 0.1},
                        ]
                    },
                    "featuresFoundCount": 1,
                    "evalSetMetrics": [
                        {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
                        {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
                    ],
                }
            ],
            "validationImportantProviders": [],
            "createdAt": 1633302145414,
        },
    )
    requests_mock.get(
        url + "/public/api/v2/search/features/432",
        json={
            "providerFeatures": [
                {"name": "feature", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"},
            ],
            "etalonFeatures": [
                {"name": "SystemRecordId_473310000", "importance": 1.0, "matchedInPercent": 100.0}
            ]
        },
    )
    requests_mock.get(
        url + "/public/api/v2/search/rawfeatures/321",
        json={"adsSearchTaskFeaturesDTO": [{"searchType": "INITIAL", "adsSearchTaskFeaturesId": "333"}]},
    )
    with open(path_to_mock_features, "rb") as f:
        buffer = f.read()
        requests_mock.get(url + "/public/api/v2/search/rawfeatures/333/file", content=buffer)

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
        accurate_model=True,
        endpoint=url,
        api_key="fake_api_key",
    )

    enriched_train_features = enricher.fit_transform(
        train_features, train_target, eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)]
    )
    assert enriched_train_features.shape == (10000, 4)

    metrics = enricher.get_metrics()
    expected_metrics = pd.DataFrame(
        [
            {"match rate": 99.9, "auc": 0.66, "uplift": 0.1},
            {"match rate": 100.0, "auc": 0.5},
            {"match rate": 99.0, "auc": 0.77},
        ],
        index=["train", "eval 1", "eval 2"],
    )

    assert metrics is not None and metrics.equals(expected_metrics)

    assert enricher.feature_names_ == ["feature"]
    assert enricher.feature_importances_ == [10.1]
    assert len(enricher.features_info) == 2
    first_feature_info = enricher.features_info.iloc[0]
    assert first_feature_info["feature_name"] == "SystemRecordId_473310000"
    assert first_feature_info["shap_value"] == 1.0
    second_feature_info = enricher.features_info.iloc[1]
    assert second_feature_info["feature_name"] == "feature"
    assert second_feature_info["shap_value"] == 10.1


def test_features_enricher_fit_transform_runtime_parameters(requests_mock: Mocker):
    url = "http://fake_url2"
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/binary/mock_features.csv.gz"
    )
    requests_mock.post(url + "/private/api/v2/security/refresh_access_token", json={"access_token": "123"})
    requests_mock.post(
        url + "/public/api/v2/search/initial",
        json={
            "fileUploadId": "123",
            "searchTaskId": "initialSearchTaskId",
            "searchType": "INITIAL",
            "status": "SUBMITTED",
            "extractFeatures": "true",
            "returnScores": "false",
            "createdAt": 1633302145414,
        },
    )
    requests_mock.get(
        url + "/public/api/v2/search/initialSearchTaskId",
        json={
            "fileUploadTaskId": "123",
            "searchTaskId": "initialSearchTaskId",
            "searchTaskStatus": "COMPLETED",
            "featuresFoundCount": 1,
            "providersCheckedCount": 1,
            "importantProvidersCount": 1,
            "importantFeaturesCount": 1,
            "importantProviders": [
                {
                    "adsSearchTaskId": "432",
                    "searchTaskId": "initialSearchTaskId",
                    "searchType": "INITIAL",
                    "taskStatus": "COMPLETED",
                    "providerName": "Provider-123456",
                    "providerId": "123456",
                    "providerQuality": {
                        "metrics": [
                            {"code": "HIT_RATE", "value": 99.9},
                            {"code": "AUC", "value": 0.66},
                            {"code": "UPLIFT", "value": 0.1},
                        ]
                    },
                    "featuresFoundCount": 1,
                    "evalSetMetrics": [
                        {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
                        {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
                    ],
                }
            ],
            "validationImportantProviders": [],
            "createdAt": 1633302145414,
        },
    )
    requests_mock.get(
        url + "/public/api/v2/search/features/432",
        json={
            "providerFeatures": [
                {"name": "feature", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"},
            ],
            "etalonFeatures": [
                {"name": "SystemRecordId_473310000", "importance": 1.0, "matchedInPercent": 100.0}
            ]
        },
    )
    requests_mock.get(
        url + "/public/api/v2/search/rawfeatures/initialSearchTaskId",
        json={"adsSearchTaskFeaturesDTO": [{"searchType": "INITIAL", "adsSearchTaskFeaturesId": "333"}]},
    )
    with open(path_to_mock_features, "rb") as f:
        buffer = f.read()
        requests_mock.get(url + "/public/api/v2/search/rawfeatures/333/file", content=buffer)

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
        accurate_model=True,
        endpoint=url,
        api_key="fake_api_key",
        runtime_parameters=RuntimeParameters(properties={"runtimeProperty1": "runtimeValue1"}),
    )
    assert enricher.runtime_parameters is not None

    enricher.fit(
        train_features, train_target, eval_set=[(eval1_features, eval1_target), (eval2_features, eval2_target)]
    )

    fit_req = None
    initial_search_url = "http://fake_url2/public/api/v2/search/initial"
    for elem in requests_mock.request_history:
        if elem.url == initial_search_url:
            fit_req = elem

    # TODO: can be better with
    #  https://metareal.blog/en/post/2020/05/03/validating-multipart-form-data-with-requests-mock/
    # It's do-able to parse req with cgi module and verify contents
    assert fit_req is not None
    assert "runtimeProperty1" in str(fit_req.body)
    assert "runtimeValue1" in str(fit_req.body)

    requests_mock.post(
        url + "/public/api/v2/search/validation?initialSearchTaskId=initialSearchTaskId",
        json={
            "fileUploadId": "validation_fileUploadId",
            "searchTaskId": "validation_searchTaskId",
            "searchType": "VALIDATION",
            "status": "SUBMITTED",
            "extractFeatures": "true",
            "returnScores": "false",
            "createdAt": 1633302145414,
        },
    )

    requests_mock.get(
        url + "/public/api/v2/search/initialSearchTaskId",
        json={
            "fileUploadTaskId": "validation_fileUploadTaskId",
            "searchTaskId": "validation_searchTaskId",
            "searchTaskStatus": "COMPLETED",
            "featuresFoundCount": 1,
            "providersCheckedCount": 1,
            "importantProvidersCount": 1,
            "importantFeaturesCount": 1,
            "importantProviders": [
                {
                    "adsSearchTaskId": "adsSearchTaskId_initial",
                    "searchTaskId": "321",
                    "searchType": "INITIAL",
                    "taskStatus": "COMPLETED",
                    "providerName": "Provider-123456",
                    "providerId": "123456",
                    "providerQuality": {
                        "metrics": [
                            {"code": "HIT_RATE", "value": 99.9},
                            {"code": "AUC", "value": 0.66},
                            {"code": "UPLIFT", "value": 0.1},
                        ]
                    },
                    "featuresFoundCount": 1,
                    "evalSetMetrics": [
                        {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
                        {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
                    ],
                }
            ],
            "validationImportantProviders": [
                {
                    "adsSearchTaskId": "adsSearchTaskIdЗ_validation",
                    "searchTaskId": "validation_searchTaskId",
                    "searchType": "VALIDATION",
                    "taskStatus": "VALIDATION_COMPLETED",
                    "providerName": "Provider-123456",
                    "providerId": "123456",
                    "providerQuality": {
                        "metrics": [
                            {"code": "HIT_RATE", "value": 99.9},
                            {"code": "AUC", "value": 0.66},
                            {"code": "UPLIFT", "value": 0.1},
                        ]
                    },
                    "featuresFoundCount": 1,
                    "evalSetMetrics": [
                        {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
                        {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
                    ],
                }
            ],
            "createdAt": 1633302145414,
        },
    )

    requests_mock.get(
        url + "/public/api/v2/search/rawfeatures/validation_searchTaskId",
        json={"adsSearchTaskFeaturesDTO": [{"searchType": "VALIDATION", "adsSearchTaskFeaturesId": "333"}]},
    )

    transformed = enricher.transform(train_features)

    transform_req = None
    transform_url = "http://fake_url2/public/api/v2/search/validation?initialSearchTaskId=initialSearchTaskId"
    for elem in requests_mock.request_history:
        if elem.url == transform_url:
            transform_req = elem

    assert transform_req is not None
    assert "runtimeProperty1" in str(transform_req.body)
    assert "runtimeValue1" in str(transform_req.body)

    assert transformed.shape == (10000, 4)
