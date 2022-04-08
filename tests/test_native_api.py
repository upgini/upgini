import os

import pandas as pd

from upgini import Dataset, FileColumnMeaningType
from upgini.http import UPGINI_API_KEY, UPGINI_URL
from upgini.metadata import ModelTaskType
from upgini.http import init_logging


def test_initial_and_validation_search(requests_mock):
    url = "https://fake_url1"
    os.environ[UPGINI_URL] = url
    os.environ[UPGINI_API_KEY] = "fake_api_token"

    requests_mock.get("https://ident.me", content="1.1.1.1".encode())
    requests_mock.get("https://api.ipify.org", content="1.1.1.1".encode())
    requests_mock.post(url + "/private/api/v2/events/send", content="Success".encode())

    requests_mock.post(url + "/private/api/v2/security/refresh_access_token", json={"access_token": "123"})

    init_logging(url)

    requests_mock.post(
        url + "/public/api/v2/search/initial",
        json={
            "fileUploadId": "123",
            "searchTaskId": "321",
            "searchType": "INITIAL",
            "status": "SUBMITTED",
            "extractFeatures": "false",
            "returnScores": "false",
            "createdAt": 1633302145414,
        },
    )
    requests_mock.post(
        url + "/public/api/v2/search/validation?initialSearchTaskId=321",
        json={
            "fileUploadId": "234",
            "searchTaskId": "432",
            "searchType": "VALIDATION",
            "status": "SUBMITTED",
            "extractFeatures": "false",
            "returnScores": "true",
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
                        "metrics": [{"code": "HIT_RATE", "value": 100}, {"code": "AUC", "value": 0.66}]
                    },
                    "featuresFoundCount": 1,
                }
            ],
            "validationImportantProviders": [
                {
                    "adsSearchTaskId": "543",
                    "searchTaskId": "432",
                    "searchType": "VALIDATION",
                    "taskStatus": "COMPLETED",
                    "providerName": "Provider-123456",
                    "providerId": "123456",
                    "providerQuality": {"metrics": [{"code": "HIT_RATE", "value": 90}, {"code": "AUC", "value": 0.55}]},
                    "featuresFoundCount": 1,
                }
            ],
            "createdAt": 1633302145414,
        },
    )

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/binary/data.csv")
    df = pd.read_csv(path)
    meaning_types = {
        "phone_num": FileColumnMeaningType.MSISDN,
        "rep_date": FileColumnMeaningType.DATE,
        "target": FileColumnMeaningType.TARGET,
    }
    search_keys = [("phone_num", "rep_date")]
    etalon = Dataset("my_dataset", "description", df=df, model_task_type=ModelTaskType.BINARY)
    etalon.meaning_types = meaning_types
    etalon.search_keys = search_keys

    meta = etalon.calculate_metrics()
    assert meta.task_type == ModelTaskType.BINARY

    search_task = etalon.search(return_scores=True)

    initial_metadata = search_task.initial_metadata()

    assert initial_metadata.loc[0, "provider_id"] == "123456"
    assert initial_metadata.loc[0, "model_id"] == "432"
    assert initial_metadata.loc[0, "hit_rate"] == 100
    assert initial_metadata.loc[0, "roc-auc"] == 0.66

    assert search_task.initial_max_auc() is not None
    # pyright: reportOptionalSubscript=false
    assert search_task.initial_max_auc()["provider_id"] == "123456"
    assert search_task.initial_max_auc()["value"] == 0.66
    assert search_task.initial_max_uplift() is None

    validation_dataset = Dataset("my_validation", df=df)
    validation_dataset.meaning_types = meaning_types
    validation_dataset.search_keys = search_keys

    validation_search = search_task.validation(validation_dataset)

    validation_metadata = validation_search.validation_metadata()

    assert validation_metadata.loc[0, "provider_id"] == "123456"
    assert validation_metadata.loc[0, "model_id"] == "543"
    assert validation_metadata.loc[0, "hit_rate"] == 90
    assert validation_metadata.loc[0, "roc-auc"] == 0.55

    assert validation_search.validation_max_auc() is not None
    assert validation_search.validation_max_auc()["provider_id"] == "123456"
    assert validation_search.validation_max_auc()["value"] == 0.55
    assert validation_search.validation_max_uplift() is None
