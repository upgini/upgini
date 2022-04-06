import logging
import os
import time
from functools import lru_cache
from http.client import HTTPConnection
from json import dumps
from typing import Optional
from urllib.parse import urljoin

import requests
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger
from requests.exceptions import RequestException

from upgini.errors import HttpError, UnauthorizedError
from upgini.metadata import FileMetadata, FileMetrics, SearchCustomization
from upgini.utils.track_info import get_track_metrics

try:
    from importlib_metadata import version

    __version__ = version("upgini")
except ImportError:
    try:
        from importlib.metadata import version  # type: ignore

        __version__ = version("upgini")
    except ImportError:
        __version__ = "Upgini wasn't installed"

UPGINI_URL: str = "UPGINI_URL"
UPGINI_API_KEY: str = "UPGINI_API_KEY"


def debug_requests_on():
    """Switches on logging of the requests module."""
    HTTPConnection.debuglevel = 1

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True


class FileColumn:
    def __init__(self, column_json: dict):
        self.index = column_json["columnIndex"]
        if "fileColumnMeaningType" in column_json:
            self.meaning_type = column_json["fileColumnMeaningType"]
        self.data_type = column_json["fileColumnDataType"]
        self.name = column_json["columnName"]
        self.sample = column_json["sampleValues"]

    def to_json(self) -> dict:
        return {
            "columnIndex": self.index,
            "fileColumnMeaningType": self.meaning_type,
            "fileColumnDataType": self.data_type,
            "initialColumnName": self.name,
        }

    def __repr__(self) -> str:
        return self.name


class FileUploadResponse:
    def __init__(self, response_json: dict):
        self.file_upload_task_id = response_json["fileUploadTaskId"]
        self.file_name = response_json["fileName"]
        self.columns = []
        for column_json in response_json["fileColumns"]["columns"]:
            self.columns.append(FileColumn(column_json))


class SearchTaskResponse:
    def __init__(self, response: dict):
        self.file_upload_id = response["fileUploadId"]
        self.search_task_id = response["searchTaskId"]
        self.initial_search_task_id = response.get("initialSearchTaskId")
        self.search_type = response["searchType"]
        self.status = response["status"]
        self.extract_features = response["extractFeatures"]
        self.return_scores = response["returnScores"]
        self.created_at = response["createdAt"]


class ModelEvalSet(BaseModel):
    eval_set_index: int
    hit_rate: float
    uplift: Optional[float]
    auc: Optional[float]
    gini: Optional[float]
    rmse: Optional[float]
    accuracy: Optional[float]


class ProviderTaskSummary:
    def __init__(self, response: dict):
        self.ads_search_task_id = response["adsSearchTaskId"]
        self.search_task_id = response["searchTaskId"]
        self.search_type = response["searchType"]
        self.status = response["taskStatus"]
        self.provider_name = response["providerName"]
        self.provider_id = response["providerId"]
        self.error_message = response.get("errorMessage")
        self.metrics = {metric["code"]: metric["value"] for metric in response["providerQuality"]["metrics"]}
        self.features_found_count = response["featuresFoundCount"]
        if "evalSetMetrics" in response.keys() is not None:
            self.eval_set_metrics = [ModelEvalSet.parse_obj(metrics) for metrics in response["evalSetMetrics"]]
        else:
            self.eval_set_metrics = None
        # providerConfusionMatrix
        # charts


class SearchTaskSummary:
    def __init__(self, response: dict):
        self.search_task_id = response["searchTaskId"]
        self.file_upload_id = response["fileUploadTaskId"]
        self.status = response["searchTaskStatus"]
        self.features_found = response["featuresFoundCount"]
        self.providers_checked = response["providersCheckedCount"]
        self.important_providers_count = response["importantProvidersCount"]
        self.important_features_count = response["importantFeaturesCount"]
        self.initial_important_providers = [
            ProviderTaskSummary(provider_response) for provider_response in response["importantProviders"]
        ]
        self.validation_important_providers = [
            ProviderTaskSummary(provider_response) for provider_response in response["validationImportantProviders"]
        ]
        self.created_at = response["createdAt"]
        # performanceMetrics


class LogEvent(BaseModel):
    source: str
    tags: str
    service: str
    hostname: str
    message: str


class _RestClient:
    SERVICE_ROOT = "public/api/v1/"
    SERVICE_ROOT_V2 = "public/api/v2/"

    REFRESH_TOKEN_URI_FMT = "private/api/v2/security/refresh_access_token"

    # V2
    CHECK_UPLOADED_FILE_URL_FMT_V2 = SERVICE_ROOT_V2 + "search/check-file?fileUploadId={0}"
    INITIAL_SEARCH_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/initial"
    INITIAL_SEARCH_WITHOUT_UPLOAD_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/initial-without-upload?fileUploadId={0}"
    VALIDATION_SEARCH_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/validation?initialSearchTaskId={0}"
    VALIDATION_SEARCH_WITHOUT_UPLOAD_URI_FMT_V2 = (
        SERVICE_ROOT_V2 + "search/validation-without-upload?fileUpload_id={0}&initialSearchTask={1}"
    )
    SEARCH_TASK_SUMMARY_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/{0}"
    SEARCH_TASK_FEATURES_META_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/features/{0}"
    STOP_SEARCH_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/{0}/stop"
    SEARCH_MODELS_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/models/{0}"
    SEARCH_SCORES_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/scores/{0}"
    SEARCH_FEATURES_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/rawfeatures/{0}"
    SEARCH_MODEL_FILE_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/models/{0}/file"
    SEARCH_SCORES_FILE_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/scores/{0}/file"
    SEARCH_FEATURES_FILE_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/rawfeatures/{0}/file"
    SEARCH_FILE_METADATA_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/{0}/metadata"

    UPLOAD_USER_ADS_URI = SERVICE_ROOT + "ads/upload"
    SEND_LOG_EVENT_URI = "private/api/v2/events/send"

    ACCESS_TOKEN_HEADER_NAME = "Authorization"
    CONTENT_TYPE_HEADER_NAME = "Content-Type"
    CONTENT_TYPE_HEADER_VALUE_JSON = "application/json;charset=UTF8"
    CHUNK_SIZE = 0x200000
    DEFAULT_OWNER = "Python SDK"
    USER_AGENT_HEADER_NAME = "User-Agent"
    USER_AGENT_HEADER_VALUE = "pyupgini/" + __version__

    _access_token: Optional[str] = None

    def __init__(self, service_endpoint, refresh_token):
        # debug_requests_on()
        self._service_endpoint = service_endpoint
        self._refresh_token = refresh_token
        self._refresh_access_token()

    def _refresh_access_token(self) -> str:
        api_path = self.REFRESH_TOKEN_URI_FMT
        response = requests.post(
            url=urljoin(self._service_endpoint, api_path),
            json={"refreshToken": self._refresh_token},
        )
        if response.status_code >= 400:
            raise HttpError(response.text, response.status_code)
        self._access_token = response.json()["access_token"]
        return self._access_token

    def _get_access_token(self) -> str:
        if self._access_token is not None:
            return self._access_token
        else:
            return self._refresh_access_token()

    def _with_unauth_retry(self, request):
        try:
            return request()
        except RequestException as e:
            print(f"Connection error: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            return self._with_unauth_retry(request)
        except UnauthorizedError:
            self._refresh_access_token()
            return request()

    def initial_search_v2(
        self,
        file_path: str,
        metadata: FileMetadata,
        metrics: FileMetrics,
        search_customization: Optional[SearchCustomization],
    ) -> SearchTaskResponse:
        api_path = self.INITIAL_SEARCH_URI_FMT_V2

        def open_and_send():
            with open(file_path, "rb") as file:
                files = {
                    "metadata": ("metadata.json", metadata.json(exclude_none=True).encode(), "application/json"),
                    "metrics": ("metrics.json", metrics.json(exclude_none=True).encode(), "application/json"),
                    "file": (metadata.name, file, "application/octet-stream"),
                }
                if search_customization is not None:
                    files["customization"] = (
                        "customization.json",
                        search_customization.json(exclude_none=True).encode(),
                        "application/json",
                    )
                files["tracking"] = ("tracking.json", dumps(get_track_metrics()).encode(), "application/json")

                return self._send_post_file_req_v2(api_path, files)

        response = self._with_unauth_retry(lambda: open_and_send())
        return SearchTaskResponse(response)

    def check_uploaded_file_v2(self, file_upload_id: str, metadata: FileMetadata) -> bool:
        api_path = self.CHECK_UPLOADED_FILE_URL_FMT_V2.format(file_upload_id)
        response = self._with_unauth_retry(lambda: self._send_post_req(api_path, metadata.json(exclude_none=True)))
        return bool(response)

    def initial_search_without_upload_v2(
        self,
        file_upload_id: str,
        metadata: FileMetadata,
        metrics: FileMetrics,
        search_customization: Optional[SearchCustomization],
    ) -> SearchTaskResponse:
        api_path = self.INITIAL_SEARCH_WITHOUT_UPLOAD_URI_FMT_V2.format(file_upload_id)
        files = {
            "metadata": ("metadata.json", metadata.json(exclude_none=True).encode(), "application/json"),
            "metrics": ("metrics.json", metrics.json(exclude_none=True).encode(), "application/json"),
        }
        if search_customization is not None:
            files["customization"] = search_customization.json(exclude_none=True).encode()
        response = self._with_unauth_retry(lambda: self._send_post_file_req_v2(api_path, files))
        return SearchTaskResponse(response)

    def validation_search_v2(
        self,
        file_path: str,
        initial_search_task_id: str,
        metadata: FileMetadata,
        metrics: FileMetrics,
        search_customization: Optional[SearchCustomization],
    ) -> SearchTaskResponse:
        api_path = self.VALIDATION_SEARCH_URI_FMT_V2.format(initial_search_task_id)

        def open_and_send():
            with open(file_path, "rb") as file:
                files = {
                    "metadata": ("metadata.json", metadata.json(exclude_none=True).encode(), "application/json"),
                    "metrics": ("metrics.json", metrics.json(exclude_none=True).encode(), "application/json"),
                    "file": (metadata.name, file, "application/octet-stream"),
                }
                if search_customization is not None:
                    files["customization"] = (
                        "customization.json",
                        search_customization.json(exclude_none=True).encode(),
                        "application/json",
                    )
                files["tracking"] = ("ide", dumps(get_track_metrics()).encode(), "application/json")

                return self._send_post_file_req_v2(api_path, files)

        response = self._with_unauth_retry(lambda: open_and_send())
        return SearchTaskResponse(response)

    def validation_search_without_upload_v2(
        self,
        file_upload_id: str,
        initial_search_task_id: str,
        metadata: FileMetadata,
        metrics: FileMetrics,
        search_customization: Optional[SearchCustomization],
    ) -> SearchTaskResponse:
        api_path = self.VALIDATION_SEARCH_WITHOUT_UPLOAD_URI_FMT_V2.format(file_upload_id, initial_search_task_id)
        files = {
            "metadata": ("metadata.json", metadata.json(exclude_none=True).encode(), "application/json"),
            "metrics": ("metrics.json", metrics.json(exclude_none=True).encode(), "application/json"),
        }
        if search_customization is not None:
            files["customization"] = search_customization.json(exclude_none=True).encode()
        response = self._with_unauth_retry(lambda: self._send_post_file_req_v2(api_path, files))
        return SearchTaskResponse(response)

    def search_task_summary_v2(self, search_task_id: str) -> SearchTaskSummary:
        api_path = self.SEARCH_TASK_SUMMARY_URI_FMT_V2.format(search_task_id)
        response = self._with_unauth_retry(lambda: self._send_get_req(api_path))
        return SearchTaskSummary(response)

    def stop_search_task_v2(self, search_task_id: str):
        api_path = self.STOP_SEARCH_URI_FMT_V2.format(search_task_id)
        self._with_unauth_retry(lambda: self._send_post_req(api_path))

    def get_search_features_meta_v2(self, provider_search_task_id: str):
        api_path = self.SEARCH_TASK_FEATURES_META_URI_FMT_V2.format(provider_search_task_id)
        return self._with_unauth_retry(lambda: self._send_get_req(api_path))

    def get_search_models_v2(self, search_task_id):
        api_path = self.SEARCH_MODELS_URI_FMT_V2.format(search_task_id)
        return self._with_unauth_retry(lambda: self._send_get_req(api_path))

    def get_search_model_file_v2(self, trained_model_id):
        api_path = self.SEARCH_MODEL_FILE_URI_FMT_V2.format(trained_model_id)
        return self._with_unauth_retry(lambda: self._send_get_file_req(api_path))

    def get_search_scores_v2(self, search_task_id):
        api_path = self.SEARCH_SCORES_URI_FMT_V2.format(search_task_id)
        return self._with_unauth_retry(lambda: self._send_get_req(api_path))

    def get_search_scores_file_v2(self, ads_scores_id):
        api_path = self.SEARCH_SCORES_FILE_URI_FMT_V2.format(ads_scores_id)
        return self._with_unauth_retry(lambda: self._send_get_file_req(api_path))

    def get_search_features_v2(self, search_task_id):
        api_path = self.SEARCH_FEATURES_URI_FMT_V2.format(search_task_id)
        return self._with_unauth_retry(lambda: self._send_get_req(api_path))

    def get_search_features_file_v2(self, ads_features_id):
        api_path = self.SEARCH_FEATURES_FILE_URI_FMT_V2.format(ads_features_id)
        return self._with_unauth_retry(lambda: self._send_get_file_req(api_path))

    def upload_user_ads(self, file_path: str, metadata: FileMetadata):
        api_path = self.UPLOAD_USER_ADS_URI

        def open_and_send():
            with open(file_path, "rb") as file:
                files = {
                    "file": (metadata.name, file, "application/octet-stream"),
                    "metadata": ("metadata.json", metadata.json(exclude_none=True).encode(), "application/json"),
                }

                return self._send_post_file_req_v2(api_path, files)

        return self._with_unauth_retry(lambda: open_and_send())

    def get_search_file_metadata(self, search_task_id: str) -> FileMetadata:
        api_path = self.SEARCH_FILE_METADATA_URI_FMT_V2.format(search_task_id)
        response = self._with_unauth_retry(lambda: self._send_get_req(api_path))
        return FileMetadata.parse_obj(response)

    def send_log_event(self, log_event: LogEvent):
        api_path = self.SEND_LOG_EVENT_URI
        self._with_unauth_retry(
            lambda: self._send_post_req(
                api_path,
                json_data=log_event.dict(exclude_none=True),
                content_type="application/json",
                result_format="text",
                silent=True,
            )
        )

    # ---

    def _send_get_req(self, api_path):
        response = requests.get(url=urljoin(self._service_endpoint, api_path), headers=self._get_headers())

        if response.status_code >= 400:
            raise HttpError(response.text, status_code=response.status_code)

        return response.json()

    def _send_get_file_req(self, api_path):
        response = requests.get(url=urljoin(self._service_endpoint, api_path), headers=self._get_headers())

        if response.status_code >= 400:
            raise HttpError(response.text, status_code=response.status_code)

        return response.content

    def _send_post_req(
        self, api_path, json_data=None, data=None, content_type=None, result_format="json", silent=False
    ):
        response = requests.post(
            url=urljoin(self._service_endpoint, api_path),
            data=data,
            json=json_data,
            headers=self._get_headers(content_type),
        )

        if response.status_code >= 400:
            if not silent:
                logging.error(f"Failed to execute request to {api_path}: {response}")
            raise HttpError(response.text, status_code=response.status_code)

        if result_format == "json":
            return response.json()
        else:
            return response.text

    def _send_post_file_req_v2(self, api_path, files, data=None):
        response = requests.post(
            url=urljoin(self._service_endpoint, api_path),
            data=data,
            files=files,
            headers=self._get_headers(),
        )

        if response.status_code >= 400:
            raise HttpError(response.text, status_code=response.status_code)

        return response.json()

    def _get_headers(self, content_type=None):
        headers = {
            self.USER_AGENT_HEADER_NAME: self.USER_AGENT_HEADER_VALUE,
            self.ACCESS_TOKEN_HEADER_NAME: "Bearer " + self._get_access_token(),
        }
        if content_type:
            headers[self.CONTENT_TYPE_HEADER_NAME] = content_type
        return headers

    @staticmethod
    def _update_columns_request(update_columns):
        return {"columns": [x.to_json() for x in update_columns]}


@lru_cache()
def get_rest_client(backend_url: Optional[str] = None, api_token: Optional[str] = None) -> _RestClient:
    if backend_url is not None:
        url = backend_url
    elif UPGINI_URL not in os.environ:
        url = "https://search.upgini.com"
    else:
        url = os.environ[UPGINI_URL]

    if api_token is not None:
        token = api_token
    elif UPGINI_API_KEY not in os.environ:
        # Demo user api-key
        token = "aIB6TC-BcuMlvoHxRwkJhn7hI-okN-dkE6RGLrZBKDw"
    else:
        token = os.environ[UPGINI_API_KEY]

    return _RestClient(url, token)


class BackendLogHandler(logging.Handler):
    def __init__(self, rest_client: _RestClient, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rest_client = rest_client
        self.hostname = get_track_metrics()["ip"]

    def emit(self, record: logging.LogRecord) -> None:
        text = self.format(record)
        tags = get_track_metrics()
        tags["version"] = __version__
        self.rest_client.send_log_event(
            LogEvent(
                source="python",
                tags=",".join([f"{k}:{v}" for k, v in tags.items()]),
                hostname=self.hostname,
                message=text,
                service="PyLib",
            )
        )


def init_logging(backend_url: Optional[str] = None, api_token: Optional[str] = None):
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()

    root.setLevel(logging.INFO)

    rest_client = get_rest_client(backend_url, api_token)
    datadogHandler = BackendLogHandler(rest_client)
    jsonFormatter = jsonlogger.JsonFormatter("%(asctime)s %(threadName)s %(name)s %(levelname)s %(message)s")
    datadogHandler.setFormatter(jsonFormatter)
    root.addHandler(datadogHandler)
