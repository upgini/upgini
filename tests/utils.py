from random import randint
from typing import List

from requests_mock import Mocker


def mock_default_requests(requests_mock: Mocker, url: str):
    requests_mock.get("https://ident.me", content="1.1.1.1".encode())
    requests_mock.post(url + "/private/api/v2/events/send", content="Success".encode())
    requests_mock.post(url + "/private/api/v2/security/refresh_access_token", json={"access_token": "123"})


def random_id() -> str:
    return str(randint(11111, 99999))


def mock_initial_search(requests_mock: Mocker, url: str) -> str:
    search_task_id = random_id()
    requests_mock.post(
        url + "/public/api/v2/search/initial",
        json={
            "fileUploadId": random_id(),
            "searchTaskId": search_task_id,
            "searchType": "INITIAL",
            "status": "SUBMITTED",
            "extractFeatures": "true",
            "returnScores": "false",
            "createdAt": 1633302145414,
        },
    )
    return search_task_id


def mock_initial_summary(requests_mock: Mocker, url: str, search_task_id: str) -> str:
    ads_search_task_id = random_id()
    requests_mock.get(
        url + "/public/api/v2/search/" + search_task_id,
        json={
            "fileUploadTaskId": random_id(),
            "searchTaskId": search_task_id,
            "searchTaskStatus": "COMPLETED",
            "featuresFoundCount": 1,
            "providersCheckedCount": 1,
            "importantProvidersCount": 1,
            "importantFeaturesCount": 1,
            "importantProviders": [
                {
                    "adsSearchTaskId": ads_search_task_id,
                    "searchTaskId": search_task_id,
                    "searchType": "INITIAL",
                    "taskStatus": "COMPLETED",
                    "providerName": "Provider-123456",
                    "providerId": "123456",
                    "providerQuality": {
                        "metrics": [
                            {"code": "HIT_RATE", "value": 99.9},
                            # {"code": "AUC", "value": 0.66},
                            # {"code": "UPLIFT", "value": 0.1},
                        ]
                    },
                    "featuresFoundCount": 1,
                    "evalSetMetrics": [
                        # {"eval_set_index": 1, "hit_rate": 100, "auc": 0.5},
                        # {"eval_set_index": 2, "hit_rate": 99, "auc": 0.77},
                    ],
                }
            ],
            "validationImportantProviders": [],
            "createdAt": 1633302145414,
        },
    )
    return ads_search_task_id


def mock_get_metadata(requests_mock: Mocker, url: str, search_task_id: str):
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
                    "name": "systemrecordid_473310000",
                    "originalName": "SystemRecordId_473310000",
                    "dataType": "STRING",
                    "meaningType": "FEATURE",
                },
                {
                    "index": 1,
                    "name": "phone_num",
                    "originalName": "phone_num",
                    "dataType": "STRING",
                    "meaningType": "MSISDN",
                },
                {"index": 2, "name": "rep_date", "originalName": "rep_date", "dataType": "INT", "meaningType": "DATE"},
                {"index": 3, "name": "target", "originalName": "target", "dataType": "INT", "meaningType": "DATE"},
                {
                    "index": 4,
                    "name": "system_record_id",
                    "originalName": "system_record_id",
                    "dataType": "INT",
                    "meaningType": "SYSTEM_RECORD_ID",
                },
            ],
            "searchKeys": [["phone_num"], ["rep_date"], ["phone_num", "rep_date"]],
            "hierarchicalGroupKeys": [],
            "hierarchicalSubgroupKeys": [],
            "rowsCount": 15555,
        },
    )


def mock_get_features_meta(
    requests_mock: Mocker, url: str, ads_search_task_id: str, ads_features: List[dict], etalon_features: List[dict] = []
):
    requests_mock.get(
        url + "/public/api/v2/search/features/" + ads_search_task_id,
        json={
            "providerFeatures": ads_features,
            "etalonFeatures": etalon_features,
        },
    )


def mock_raw_features(requests_mock: Mocker, url: str, search_task_id: str, path_to_mock_features: str):
    ads_search_task_features_id = random_id()
    requests_mock.get(
        url + "/public/api/v2/search/rawfeatures/" + search_task_id,
        json={
            "adsSearchTaskFeaturesDTO": [
                {"searchType": "INITIAL", "adsSearchTaskFeaturesId": ads_search_task_features_id}
            ]
        },
    )
    with open(path_to_mock_features, "rb") as f:
        buffer = f.read()
        requests_mock.get(url + f"/public/api/v2/search/rawfeatures/{ads_search_task_features_id}/file", content=buffer)
