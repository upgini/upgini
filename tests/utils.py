import itertools
import json
import tempfile
from random import randint
from typing import Dict, List, Optional, Union
import uuid

import pandas as pd
from requests_mock import Mocker

from upgini.metadata import ProviderTaskMetadataV2


class RequestsCounter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1


def mock_default_requests(requests_mock: Mocker, url: str):
    requests_mock.get("https://ident.me", content=b"1.1.1.1")
    requests_mock.post(url + "/private/api/v2/events/send", content=b"Success")
    requests_mock.post(url + "/private/api/v2/security/refresh_access_token", json={"access_token": "123"})
    requests_mock.get("https://pypi.python.org/pypi/upgini/json", json={"releases": {"1.0.0": [{}]}})
    requests_mock.post(url + "/public/api/v2/search/dump-input", content=b"123")
    requests_mock.get(
        url + "/public/api/v2/user/transform-usage",
        json={"transformedRows": 0, "restRows": 12000, "limit": 12000, "hasLimit": True},
    )


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


# def _construct_metrics(
#     hit_rate: float, auc: Optional[float], rmse: Optional[float], accuracy: Optional[float], uplift: Optional[float]
# ) -> List[dict]:
#     metrics = [{"code": "HIT_RATE", "value": hit_rate}]
#     if auc is not None:
#         metrics.append({"code": "AUC", "value": auc})
#     if rmse is not None:
#         metrics.append({"code": "RMSE", "value": rmse})
#     if accuracy is not None:
#         metrics.append({"code": "ACCURACY", "value": accuracy})
#     if uplift is not None:
#         metrics.append({"code": "UPLIFT", "value": uplift})
#     return metrics


def mock_initial_summary(
    requests_mock: Mocker,
    url: str,
    search_task_id: str,
) -> str:
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
                }
            ],
            "validationImportantProviders": [],
            "createdAt": 1633302145414,
        },
    )
    return ads_search_task_id


def mock_get_metadata(
    requests_mock: Mocker,
    url: str,
    search_task_id: str,
    metadata_columns: Optional[List[Dict]] = None,
    search_keys: Optional[List[str]] = None,
):
    if metadata_columns is None:
        metadata_columns = [
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
            {"index": 3, "name": "target", "originalName": "target", "dataType": "INT", "meaningType": "TARGET"},
            {
                "index": 4,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
        ]
    if search_keys is None:
        search_keys = ["phone_num", "rep_date"]
    combined_search_keys = []
    for L in range(1, len(search_keys) + 1):
        for subset in itertools.combinations(search_keys, L):
            combined_search_keys.append(subset)
    requests_mock.get(
        url + f"/public/api/v2/search/{search_task_id}/metadata",
        json={
            "fileUploadId": "123",
            "fileMetadataId": "123",
            "name": "test",
            "description": "",
            "columns": metadata_columns,
            "searchKeys": combined_search_keys,
            "hierarchicalGroupKeys": [],
            "hierarchicalSubgroupKeys": [],
            "rowsCount": 15555,
        },
    )


def mock_get_task_metadata_v2(requests_mock: Mocker, url: str, ads_search_task_id: str, meta: ProviderTaskMetadataV2):
    requests_mock.get(url + "/public/api/v2/search/metadata-v2/" + ads_search_task_id, json=meta.dict())


def mock_get_task_metadata_v2_from_file(requests_mock: Mocker, url: str, ads_search_task_id: str, meta_path: str):
    with open(meta_path, "r") as f:
        meta = json.load(f)
        requests_mock.get(url + "/public/api/v2/search/metadata-v2/" + ads_search_task_id, json=meta)


def mock_raw_features(
    requests_mock: Mocker,
    url: str,
    search_task_id: str,
    mock_features: Union[str, pd.DataFrame],
    metrics_calculation=True,
):
    ads_search_task_features_id = random_id()
    api = (
        f"{url}/public/api/v2/search/rawfeatures/{search_task_id}"
        f"?metricsCalculation={str(metrics_calculation).lower()}"
    )
    requests_mock.get(
        api,
        json={
            "adsSearchTaskFeaturesDTO": [
                {"searchType": "INITIAL", "adsSearchTaskFeaturesId": ads_search_task_features_id}
            ]
        },
    )
    if isinstance(mock_features, str):
        with open(mock_features, "rb") as f:
            buffer = f.read()
            requests_mock.get(
                url + f"/public/api/v2/search/rawfeatures/{ads_search_task_features_id}/file", content=buffer
            )
    elif isinstance(mock_features, pd.DataFrame):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_features.to_parquet(f"{tmp_dir}/tmp.parquet")
            with open(f"{tmp_dir}/tmp.parquet", "rb") as f:
                buffer = f.read()
                requests_mock.get(
                    url + f"/public/api/v2/search/rawfeatures/{ads_search_task_features_id}/file", content=buffer
                )
    else:
        raise Exception(
            f"Unsupported type of mock features: {type(mock_features)}. Supported only string (path) or DataFrame"
        )


def mock_validation_raw_features(
    requests_mock: Mocker,
    url: str,
    validation_search_task_id: str,
    mock_features: Union[str, pd.DataFrame],
    metrics_calculation=False,
):
    ads_search_task_features_id = random_id()
    api = (
        f"{url}/public/api/v2/search/rawfeatures/{validation_search_task_id}"
        f"?metricsCalculation={str(metrics_calculation).lower()}"
    )
    requests_mock.get(
        api,
        json={
            "adsSearchTaskFeaturesDTO": [
                {"searchType": "VALIDATION", "adsSearchTaskFeaturesId": ads_search_task_features_id}
            ]
        },
    )
    if isinstance(mock_features, str):
        with open(mock_features, "rb") as f:
            buffer = f.read()
            requests_mock.get(
                url + f"/public/api/v2/search/rawfeatures/{ads_search_task_features_id}/file", content=buffer
            )
    elif isinstance(mock_features, pd.DataFrame):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_features.to_parquet(f"{tmp_dir}/tmp.parquet")
            with open(f"{tmp_dir}/tmp.parquet", "rb") as f:
                buffer = f.read()
                requests_mock.get(
                    url + f"/public/api/v2/search/rawfeatures/{ads_search_task_features_id}/file", content=buffer
                )
    else:
        raise Exception(
            f"Unsupported type of mock features: {type(mock_features)}. Supported only string (path) or DataFrame"
        )


def mock_validation_search(requests_mock: Mocker, url: str, initial_search_task_id: str) -> str:
    validation_search_task_id = random_id()
    requests_mock.post(
        url + "/public/api/v2/search/validation?initialSearchTaskId=" + initial_search_task_id,
        json={
            "fileUploadId": "validation_fileUploadId",
            "searchTaskId": validation_search_task_id,
            "searchType": "VALIDATION",
            "status": "SUBMITTED",
            "extractFeatures": "true",
            "returnScores": "false",
            "createdAt": 1633302145414,
        },
    )
    return validation_search_task_id


def mock_validation_summary(
    requests_mock: Mocker,
    url: str,
    initial_search_task_id: str,
    initial_ads_search_task_id: str,
    validation_search_task_id: str,
) -> str:
    ads_search_task_id = random_id()
    requests_mock.get(
        url + "/public/api/v2/search/" + initial_search_task_id,
        json={
            "fileUploadTaskId": "validation_fileUploadTaskId",
            "searchTaskId": initial_search_task_id,
            "searchTaskStatus": "COMPLETED",
            "featuresFoundCount": 1,
            "providersCheckedCount": 1,
            "importantProvidersCount": 1,
            "importantFeaturesCount": 1,
            "importantProviders": [
                {
                    "adsSearchTaskId": initial_ads_search_task_id,
                    "searchTaskId": initial_search_task_id,
                    "searchType": "INITIAL",
                    "taskStatus": "COMPLETED",
                }
            ],
            "validationImportantProviders": [
                {
                    "adsSearchTaskId": ads_search_task_id,
                    "searchTaskId": validation_search_task_id,
                    "searchType": "VALIDATION",
                    "taskStatus": "VALIDATION_COMPLETED",
                }
            ],
            "createdAt": 1633302145414,
        },
    )
    return ads_search_task_id


def mock_initial_and_validation_summary(
    requests_mock: Mocker,
    url: str,
    search_task_id: str,
    validation_search_task_id: str,
):
    ads_search_task_id = random_id()
    validation_ads_search_task_id = random_id()
    # metrics = _construct_metrics(hit_rate, auc, rmse, accuracy, uplift)

    req_counter = RequestsCounter()
    file_upload_task_id = random_id()

    def response(request, context):
        if req_counter.count == 0:
            req_counter.increment()
            return {
                "fileUploadTaskId": file_upload_task_id,
                "searchTaskId": search_task_id,
                "searchTaskStatus": "CREATED",
                "featuresFoundCount": 1,
                "providersCheckedCount": 1,
                "importantProvidersCount": 1,
                "importantFeaturesCount": 1,
                "importantProviders": [],
                "validationImportantProviders": [],
                "createdAt": 1633302145414,
            }
        elif req_counter.count == 1:
            req_counter.increment()
            return {
                "fileUploadTaskId": file_upload_task_id,
                "searchTaskId": search_task_id,
                "searchTaskStatus": "SUBMITTED",
                "featuresFoundCount": 1,
                "providersCheckedCount": 1,
                "importantProvidersCount": 1,
                "importantFeaturesCount": 1,
                "importantProviders": [
                    {
                        "adsSearchTaskId": ads_search_task_id,
                        "searchTaskId": search_task_id,
                        "searchType": "INITIAL",
                        "taskStatus": "SUBMITTED",
                    }
                ],
                "validationImportantProviders": [],
                "createdAt": 1633302145414,
            }
        elif req_counter.count == 2:
            req_counter.increment()
            return {
                "fileUploadTaskId": file_upload_task_id,
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
                    }
                ],
                "validationImportantProviders": [],
                "createdAt": 1633302145414,
            }
        elif req_counter.count == 3:
            req_counter.increment()
            return {
                "fileUploadTaskId": "validation_fileUploadTaskId",
                "searchTaskId": search_task_id,
                "searchTaskStatus": "VALIDATION_CREATED",
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
                    }
                ],
                "validationImportantProviders": [],
                "createdAt": 1633302145414,
            }
        elif req_counter.count == 4:
            req_counter.increment()
            return {
                "fileUploadTaskId": "validation_fileUploadTaskId",
                "searchTaskId": search_task_id,
                "searchTaskStatus": "VALIDATION_SUBMITTED",
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
                    }
                ],
                "validationImportantProviders": [
                    {
                        "adsSearchTaskId": validation_ads_search_task_id,
                        "searchTaskId": validation_search_task_id,
                        "searchType": "VALIDATION",
                        "taskStatus": "SUBMITTED",
                    }
                ],
                "createdAt": 1633302145414,
            }
        else:
            req_counter.increment()
            return {
                "fileUploadTaskId": "validation_fileUploadTaskId",
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
                    }
                ],
                "validationImportantProviders": [
                    {
                        "adsSearchTaskId": validation_ads_search_task_id,
                        "searchTaskId": validation_search_task_id,
                        "searchType": "VALIDATION",
                        "taskStatus": "VALIDATION_COMPLETED",
                    }
                ],
                "createdAt": 1633302145414,
            }

    requests_mock.get(
        url + "/public/api/v2/search/" + search_task_id,
        json=response,
    )
    return ads_search_task_id


def mock_initial_progress(
    requests_mock: Mocker,
    url: str,
    search_task_id: str,
):
    req_counter = RequestsCounter()

    def response(request, context):
        if req_counter.count == 0:
            req_counter.increment()
            return {
                "currentStage": "CREATING_FIT",
                "percent": 4.0,
            }
        elif req_counter.count == 1:
            req_counter.increment()
            return {
                "currentStage": "MATCHING",
                "percent": 6.0,
            }
        elif req_counter.count == 2:
            req_counter.increment()
            return {
                "currentStage": "SEARCHING",
                "percent": 40.0,
            }
        else:
            req_counter.increment()
            return {
                "currentStage": "GENERATING_REPORT",
                "percent": 97.0,
            }

    requests_mock.get(f"{url}/public/api/v2/search/{search_task_id}/progress", json=response)


def mock_validation_progress(requests_mock: Mocker, url: str, validation_search_task_id: str):
    req_counter = RequestsCounter()

    def response(request, context):
        if req_counter.count == 0:
            req_counter.increment()
            return {
                "currentStage": "CREATING_TRANSFORM",
                "percent": 4.0,
            }
        elif req_counter.count == 1:
            req_counter.increment()
            return {
                "currentStage": "ENRICHING",
                "percent": 6.0,
            }
        else:
            req_counter.increment()
            return {
                "currentStage": "DOWNLOADING",
                "percent": 97.0,
            }

    requests_mock.get(f"{url}/public/api/v2/search/{validation_search_task_id}/progress", json=response)


def mock_target_outliers(requests_mock: Mocker, url: str, search_task_id: str):
    url = f"{url}/public/api/v2/search/target-outliers/{search_task_id}"
    outlier_id = uuid.uuid4()
    response = {
        "adsSearchTaskTargetOutliersDTO": [{
            "adsSearchTaskTargetOutliersId": str(outlier_id)
        }]
    }
    requests_mock.get(url, json=response)
    return outlier_id


def mock_target_outliers_file(requests_mock: Mocker, url: str, outlier_id: str, outliers: Union[str, pd.DataFrame]):
    api_path = f"{url}/public/api/v2/search/target-outliers/{outlier_id}/file"
    if isinstance(outliers, str):
        with open(outliers, "rb") as f:
            buffer = f.read()
            requests_mock.get(api_path, content=buffer)
    elif isinstance(outliers, pd.DataFrame):
        with tempfile.TemporaryDirectory() as tmp_dir:
            outliers.to_parquet(f"{tmp_dir}/tmp.parquet")
            with open(f"{tmp_dir}/tmp.parquet", "rb") as f:
                buffer = f.read()
                requests_mock.get(api_path, content=buffer)
    else:
        raise Exception(
            f"Unsupported type of mock target outliers: {type(outliers)}. Supported only string (path) or DataFrame"
        )