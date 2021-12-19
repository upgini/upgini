import os

import pandas as pd

from upgini import FeaturesEnricher, SearchKey


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
    assert enriched_train_features.shape == (10000, 5)

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
