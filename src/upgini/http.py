import base64
import copy
import datetime
import hashlib
import logging
import os
import random
import socket
import threading
import time
from enum import Enum
from functools import lru_cache
from http.client import HTTPConnection
from json import dumps
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import jwt
import pandas as pd
import requests
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger
from requests.exceptions import RequestException

from upgini.errors import (
    HttpError,
    UnauthorizedError,
    UpginiConnectionError,
    ValidationError,
)
from upgini.metadata import (
    FileColumnMeaningType,
    FileMetadata,
    FileMetrics,
    ProviderTaskMetadataV2,
    SearchCustomization,
)
from upgini.resource_bundle import bundle
from upgini.utils.track_info import get_track_metrics

try:
    from importlib_metadata import version  # type: ignore

    __version__ = version("upgini")
except ImportError:
    try:
        from importlib.metadata import version  # type: ignore

        __version__ = version("upgini")
    except ImportError:
        __version__ = "Upgini wasn't installed"

UPGINI_URL: str = "UPGINI_URL"
UPGINI_API_KEY: str = "UPGINI_API_KEY"
DEMO_API_KEY: str = "Aa4BPwGFbn1zNEXIkZ-NbhsRk0ricN6puKuga1-O5lM"
TRACK_METRICS_TIMEOUT_SECONDS: int = 10

refresh_token_lock = threading.Lock()


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


# class ModelEvalSet(BaseModel):
#     eval_set_index: int
#     hit_rate: float
#     uplift: Optional[float]
#     auc: Optional[float]
#     gini: Optional[float]
#     rmse: Optional[float]
#     accuracy: Optional[float]


class ProviderTaskSummary:
    def __init__(self, response: dict):
        self.ads_search_task_id = response["adsSearchTaskId"]
        self.search_task_id = response["searchTaskId"]
        self.status = response["taskStatus"]
        self.error_message = response.get("errorMessage")
        self.unused_features_for_generation = response.get("unusedFeaturesForGeneration")


class SearchTaskSummary:
    def __init__(self, response: dict):
        self.search_task_id = response["searchTaskId"]
        self.file_upload_id = response["fileUploadTaskId"]
        self.status = response["searchTaskStatus"]
        # self.features_found = response["featuresFoundCount"]
        # self.providers_checked = response["providersCheckedCount"]
        # self.important_providers_count = response["importantProvidersCount"]
        # self.important_features_count = response["importantFeaturesCount"]
        self.initial_important_providers = [
            ProviderTaskSummary(provider_response) for provider_response in response["importantProviders"]
        ]
        self.validation_important_providers = [
            ProviderTaskSummary(provider_response) for provider_response in response["validationImportantProviders"]
        ]
        self.created_at = response["createdAt"]


class SearchProgress:
    def __init__(self, *args):
        # (response: dict)
        if len(args) == 1 and isinstance(args[0], dict):
            response: Dict[str, Any] = args[0]
            self.stage = response["currentStage"]
            self.percent = float(response["percent"])
            self.eta: Optional[int] = None
            self.error = response.get("errorCode")
            self.error_message = response.get("errorMessage")
        # (percent: float, stage: ProgressStage)
        elif len(args) == 2 and isinstance(args[0], float) and isinstance(args[1], ProgressStage):
            self.percent = args[0]
            self.stage = args[1].value
            self.eta: Optional[int] = None
            self.error = None
            self.error_message = None
        # (percent: float, stage: ProgressStage, eta: int)
        elif (
            len(args) == 3
            and isinstance(args[0], float)
            and isinstance(args[1], ProgressStage)
            and isinstance(args[2], float)
        ):
            self.percent = args[0]
            self.stage = args[1].value
            seconds_left = args[2]
            self.eta = int(seconds_left * (100 - self.percent) / self.percent)
            self.error = None
            self.error_message = None
        else:
            raise RuntimeError("Unsupported arguments for SearchProgress constructor")

    def recalculate_eta(self, seconds_left: int):
        self.eta = int(seconds_left * (100 - self.percent) / self.percent)

    def update_eta(self, seconds: int):
        # temporary workaround - add 5 minutes if time is over
        if seconds <= 0:
            self.eta = 5 * 60
        else:
            self.eta = seconds

    def eta_time(self) -> str:
        return str(datetime.timedelta(seconds=self.eta))

    def to_progress_bar(self) -> Tuple[int, str]:
        text = bundle.get(self.stage)
        eta = ""
        if self.eta is not None:
            eta = f"Approximately {self.eta_time()} remaining"
        return (int(self.percent), text, eta)

    def copy(self) -> "SearchProgress":
        return copy.deepcopy(self)


class ProgressStage(Enum):
    START_FIT = "START_FIT"
    START_TRANSFORM = "START_TRANSFORM"
    CREATING_FIT = "CREATING_FIT"
    CREATING_TRANSFORM = "CREATING_TRANSFORM"
    MATCHING = "MATCHING"
    AUTOFE = "AUTOFE"
    SEARCHING = "SEARCHING"
    ENRICHING = "ENRICHING"
    GENERATING_REPORT = "GENERATING_REPORT"
    DOWNLOADING = "DOWNLOADING"
    RETRIEVING_CACHE = "RETRIEVING_CACHE"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class LogEvent(BaseModel):
    source: str
    tags: str
    service: str
    hostname: str
    message: str


class _RestClient:
    PROD_BACKEND_URL = "https://search.upgini.com"

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
    SEARCH_TASK_PROGRESS_URI_FMT = SERVICE_ROOT_V2 + "search/{0}/progress"
    STOP_SEARCH_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/{0}/stop"
    SEARCH_MODELS_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/models/{0}"
    SEARCH_SCORES_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/scores/{0}"
    SEARCH_FEATURES_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/rawfeatures/{0}?metricsCalculation={1}"
    SEARCH_TARGET_OUTLIERS_URI_FMT = SERVICE_ROOT_V2 + "search/target-outliers/{0}"
    SEARCH_MODEL_FILE_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/models/{0}/file"
    SEARCH_SCORES_FILE_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/scores/{0}/file"
    SEARCH_FEATURES_FILE_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/rawfeatures/{0}/file?metricsCalculation={1}"
    SEARCH_TARGET_OUTLIERS_FILE_URI_FMT = SERVICE_ROOT_V2 + "search/target-outliers/{0}/file"
    SEARCH_FILE_METADATA_URI_FMT_V2 = SERVICE_ROOT_V2 + "search/{0}/metadata"
    SEARCH_TASK_METADATA_FMT_V3 = SERVICE_ROOT_V2 + "search/metadata-v2/{0}"
    SEARCH_DUMP_INPUT_FMT_V2 = SERVICE_ROOT_V2 + "search/dump-input"

    UPLOAD_USER_ADS_URI = SERVICE_ROOT + "ads/upload"
    SEND_LOG_EVENT_URI = "private/api/v2/events/send"

    REGISTER_ADS_URI = "private/api/v2/ads/register"
    ACTIVATE_ADS_URI = "private/api/v2/ads/activate"
    DEACTIVATE_ADS_URI = "private/api/v2/ads/deactivate"
    TOGGLE_ADS_URI_FMT = "private/api/v2/ads/{0}/toggle"
    DELETE_ADS_URI_FMT = "private/api/v2/ads/{0}"
    POLL_ADS_MANAGEMENT_STATUS_URI_FMT = "private/api/v2/ads/management-task/{0}"
    GET_ADS_DESCRIPTION_URI_FMT = "private/api/v2/ads/{0}"
    GET_ALL_ADS_DESCRIPTIONS_URI = "private/api/v2/ads/descriptions"
    GET_ACTIVE_ADS_DEFINITIONS_URI = "private/api/v2/ads/definitions"

    ACCESS_TOKEN_HEADER_NAME = "Authorization"
    CONTENT_TYPE_HEADER_NAME = "Content-Type"
    CONTENT_TYPE_HEADER_VALUE_JSON = "application/json;charset=UTF8"
    TRACE_ID_HEADER_NAME = "Trace-Id"
    CHUNK_SIZE = 0x200000
    DEFAULT_OWNER = "Python SDK"
    USER_AGENT_HEADER_NAME = "User-Agent"
    USER_AGENT_HEADER_VALUE = "pyupgini/" + __version__
    SEARCH_KEYS_HEADER_NAME = "Search-Keys"

    def __init__(self, service_endpoint, refresh_token, silent_mode=False):
        # debug_requests_on()
        self._service_endpoint = service_endpoint
        self._refresh_token = refresh_token
        self.silent_mode = silent_mode
        self._access_token = self._refresh_access_token()
        # self._access_token: Optional[str] = None  # self._refresh_access_token()
        self.last_refresh_time = time.time()

    def _refresh_access_token(self, retry_attempts_rest=5) -> str:
        api_path = self.REFRESH_TOKEN_URI_FMT
        try:
            response = requests.post(
                url=urljoin(self._service_endpoint, api_path),
                json={"refreshToken": self._refresh_token},
            )
            if response.status_code >= 400:
                raise HttpError(response.text, response.status_code)
            self._access_token = response.json()["access_token"]
            return self._access_token
        except requests.exceptions.ConnectionError:
            if retry_attempts_rest > 0:
                time.sleep(5)
                return self._refresh_access_token(retry_attempts_rest - 1)
            try:
                local_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                local_ip = ""
            try:
                requests.get("https://pypi.org")
                raise UpginiConnectionError(bundle.get("no_connection_to_upgini").format(local_ip))
            except requests.exceptions.ConnectionError:
                raise UpginiConnectionError(bundle.get("no_internet_connection").format(local_ip))

    def _syncronized_refresh_access_token(self) -> str:
        with refresh_token_lock:
            now = time.time()
            if (now - self.last_refresh_time) > 60 or self._access_token is None:
                self._access_token = self._refresh_access_token()
                self.last_refresh_time = now
        return self._access_token

    def _get_access_token(self) -> str:
        if self._access_token is not None:
            return self._access_token
        else:
            return self._syncronized_refresh_access_token()

    def _with_unauth_retry(self, request, try_number: int = 0, need_connection_retry: bool = True):
        try:
            return request()
        except RequestException as e:
            if need_connection_retry:
                print(bundle.get("connection_error_with_retry"))
                time.sleep(10)
                return self._with_unauth_retry(request)
            else:
                show_status_error()
                raise e
        except UnauthorizedError:
            self._syncronized_refresh_access_token()
            return request()
        except HttpError as e:
            if (e.status_code == 429 or e.status_code >= 500) and try_number < 3:
                time.sleep(random.randint(1, 10))
                return self._with_unauth_retry(request, try_number + 1)
            elif e.status_code == 400 and "MD5Exception".lower() in e.message.lower() and try_number < 3:
                # TODO send log
                # print(bundle.get("upload_file_checksum_fail").format(e.message))
                return self._with_unauth_retry(request, try_number + 1)
            elif e.status_code == 403:
                raise ValidationError(bundle.get("access_denied"))
            elif "more than one concurrent search request" in e.message.lower():
                raise ValidationError(bundle.get("concurrent_request"))
            else:
                show_status_error()
                raise e

    @staticmethod
    def meaning_type_by_name(name: str, metadata: FileMetadata) -> Optional[FileColumnMeaningType]:
        for c in metadata.columns:
            if c.name == name:
                return c.meaningType
        return None

    @staticmethod
    def search_keys_meaning_types(metadata: FileMetadata) -> List[str]:
        search_key_names = {key for keys in metadata.searchKeys for key in keys}
        meaning_types = [_RestClient.meaning_type_by_name(name, metadata) for name in search_key_names]
        return [meaning_type.value for meaning_type in meaning_types if meaning_type is not None]

    def dump_input_files(
        self,
        trace_id: str,
        x_path: str,
        y_path: Optional[str] = None,
        eval_x_path: Optional[str] = None,
        eval_y_path: Optional[str] = None,
    ):
        api_path = self.SEARCH_DUMP_INPUT_FMT_V2
        files = {}
        with open(x_path, "rb") as x_file:
            files["x"] = ("x.pickle", x_file, "application/octet-stream")
            if y_path:
                with open(y_path, "rb") as y_file:
                    files["y"] = ("y.pickle", y_file, "application/octet-stream")
                    if eval_x_path and eval_y_path:
                        with open(eval_x_path, "rb") as eval_x_file, open(eval_y_path, "rb") as eval_y_file:
                            files["eval_x"] = ("eval_x.pickle", eval_x_file, "application/octet-stream")
                            files["eval_y"] = ("eval_y.pickle", eval_y_file, "application/octet-stream")
                            self._with_unauth_retry(
                                lambda: self._send_post_file_req_v2(
                                    api_path, files, trace_id=trace_id, need_json_response=False
                                )
                            )
                    else:
                        self._with_unauth_retry(
                            lambda: self._send_post_file_req_v2(
                                api_path, files, trace_id=trace_id, need_json_response=False
                            )
                        )
            else:
                self._with_unauth_retry(
                    lambda: self._send_post_file_req_v2(api_path, files, trace_id=trace_id, need_json_response=False)
                )

    def initial_search_v2(
        self,
        trace_id: str,
        file_path: str,
        metadata: FileMetadata,
        metrics: FileMetrics,
        search_customization: Optional[SearchCustomization],
    ) -> SearchTaskResponse:
        api_path = self.INITIAL_SEARCH_URI_FMT_V2

        def open_and_send():
            md5_hash = hashlib.md5()
            with open(file_path, "rb") as file:
                content = file.read()
                md5_hash.update(content)
                digest = md5_hash.hexdigest()
                metadata_with_md5 = metadata.copy(update={"checksumMD5": digest})

            digest_sha256 = hashlib.sha256(
                pd.util.hash_pandas_object(pd.read_parquet(file_path, engine="fastparquet")).values
            ).hexdigest()
            metadata_with_md5 = metadata_with_md5.copy(update={"digest": digest_sha256})

            with open(file_path, "rb") as file:
                files = {
                    "metadata": (
                        "metadata.json",
                        metadata_with_md5.json(exclude_none=True).encode(),
                        "application/json",
                    ),
                    "metrics": ("metrics.json", metrics.json(exclude_none=True).encode(), "application/json"),
                    "file": (metadata_with_md5.name, file, "application/octet-stream"),
                }
                if search_customization is not None:
                    files["customization"] = (
                        "customization.json",
                        search_customization.json(exclude_none=True).encode(),
                        "application/json",
                    )
                files["tracking"] = (
                    "tracking.json",
                    dumps(get_track_metrics()).encode(),
                    "application/json",
                )
                additional_headers = {self.SEARCH_KEYS_HEADER_NAME: ",".join(self.search_keys_meaning_types(metadata))}

                return self._send_post_file_req_v2(
                    api_path, files, trace_id=trace_id, additional_headers=additional_headers
                )

        response = self._with_unauth_retry(lambda: open_and_send())
        return SearchTaskResponse(response)

    def check_uploaded_file_v2(self, trace_id: str, file_upload_id: str, metadata: FileMetadata) -> bool:
        api_path = self.CHECK_UPLOADED_FILE_URL_FMT_V2.format(file_upload_id)
        response = self._with_unauth_retry(
            lambda: self._send_post_req(api_path, trace_id, metadata.json(exclude_none=True))
        )
        return bool(response)

    def initial_search_without_upload_v2(
        self,
        trace_id: str,
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
        additional_headers = {self.SEARCH_KEYS_HEADER_NAME: ",".join(self.search_keys_meaning_types(metadata))}
        response = self._with_unauth_retry(
            lambda: self._send_post_file_req_v2(
                api_path, files, trace_id=trace_id, additional_headers=additional_headers
            )
        )
        return SearchTaskResponse(response)

    def validation_search_v2(
        self,
        trace_id: str,
        file_path: str,
        initial_search_task_id: str,
        metadata: FileMetadata,
        metrics: FileMetrics,
        search_customization: Optional[SearchCustomization],
    ) -> SearchTaskResponse:
        api_path = self.VALIDATION_SEARCH_URI_FMT_V2.format(initial_search_task_id)

        def open_and_send():
            md5_hash = hashlib.md5()
            with open(file_path, "rb") as file:
                content = file.read()
                md5_hash.update(content)
                digest = md5_hash.hexdigest()
                metadata_with_md5 = metadata.copy(update={"checksumMD5": digest})

            digest_sha256 = hashlib.sha256(
                pd.util.hash_pandas_object(pd.read_parquet(file_path, engine="fastparquet")).values
            ).hexdigest()
            metadata_with_md5 = metadata_with_md5.copy(update={"digest": digest_sha256})

            with open(file_path, "rb") as file:
                files = {
                    "metadata": (
                        "metadata.json",
                        metadata_with_md5.json(exclude_none=True).encode(),
                        "application/json",
                    ),
                    "metrics": ("metrics.json", metrics.json(exclude_none=True).encode(), "application/json"),
                    "file": (metadata_with_md5.name, file, "application/octet-stream"),
                }
                if search_customization is not None:
                    files["customization"] = (
                        "customization.json",
                        search_customization.json(exclude_none=True).encode(),
                        "application/json",
                    )
                files["tracking"] = (
                    "ide",
                    dumps(get_track_metrics()).encode(),
                    "application/json",
                )

                additional_headers = {self.SEARCH_KEYS_HEADER_NAME: ",".join(self.search_keys_meaning_types(metadata))}

                return self._send_post_file_req_v2(
                    api_path, files, trace_id=trace_id, additional_headers=additional_headers
                )

        response = self._with_unauth_retry(lambda: open_and_send())
        return SearchTaskResponse(response)

    def validation_search_without_upload_v2(
        self,
        trace_id: str,
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
        additional_headers = {self.SEARCH_KEYS_HEADER_NAME: ",".join(self.search_keys_meaning_types(metadata))}
        response = self._with_unauth_retry(
            lambda: self._send_post_file_req_v2(
                api_path, files, trace_id=trace_id, additional_headers=additional_headers
            )
        )
        return SearchTaskResponse(response)

    def search_task_summary_v2(self, trace_id: str, search_task_id: str) -> SearchTaskSummary:
        api_path = self.SEARCH_TASK_SUMMARY_URI_FMT_V2.format(search_task_id)
        response = self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))
        return SearchTaskSummary(response)

    def get_search_progress(self, trace_id: str, search_task_id: str) -> SearchProgress:
        api_path = self.SEARCH_TASK_PROGRESS_URI_FMT.format(search_task_id)
        response = self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))
        return SearchProgress(response)

    def stop_search_task_v2(self, trace_id: str, search_task_id: str):
        api_path = self.STOP_SEARCH_URI_FMT_V2.format(search_task_id)
        self._with_unauth_retry(lambda: self._send_post_req(api_path, trace_id=trace_id))

    def get_search_models_v2(self, trace_id: str, search_task_id: str):
        api_path = self.SEARCH_MODELS_URI_FMT_V2.format(search_task_id)
        return self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))

    def get_search_model_file_v2(self, trace_id: str, trained_model_id: str):
        api_path = self.SEARCH_MODEL_FILE_URI_FMT_V2.format(trained_model_id)
        return self._with_unauth_retry(lambda: self._send_get_file_req(api_path, trace_id))

    def get_search_scores_v2(self, trace_id: str, search_task_id: str):
        api_path = self.SEARCH_SCORES_URI_FMT_V2.format(search_task_id)
        return self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))

    def get_search_scores_file_v2(self, trace_id: str, ads_scores_id: str):
        api_path = self.SEARCH_SCORES_FILE_URI_FMT_V2.format(ads_scores_id)
        return self._with_unauth_retry(lambda: self._send_get_file_req(api_path, trace_id))

    def get_search_features_v2(self, trace_id: str, search_task_id: str, metrics_calculation: bool):
        api_path = self.SEARCH_FEATURES_URI_FMT_V2.format(search_task_id, str(metrics_calculation).lower())
        return self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))

    def get_search_features_file_v2(self, trace_id: str, ads_features_id: str, metrics_calculation: bool) -> bytes:
        api_path = self.SEARCH_FEATURES_FILE_URI_FMT_V2.format(ads_features_id, str(metrics_calculation).lower())
        return self._with_unauth_retry(lambda: self._send_get_file_req(api_path, trace_id))

    def get_search_target_outliners(self, trace_id: str, search_task_id: str):
        api_path = self.SEARCH_TARGET_OUTLIERS_URI_FMT.format(search_task_id)
        return self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))

    def get_search_target_outliners_file(self, trace_id: str, ads_target_outliers_id: str) -> bytes:
        api_path = self.SEARCH_TARGET_OUTLIERS_FILE_URI_FMT.format(ads_target_outliers_id)
        return self._with_unauth_retry(lambda: self._send_get_file_req(api_path, trace_id))

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

    def get_search_file_metadata(self, search_task_id: str, trace_id: str) -> FileMetadata:
        api_path = self.SEARCH_FILE_METADATA_URI_FMT_V2.format(search_task_id)
        response = self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))
        return FileMetadata.parse_obj(response)

    def get_provider_search_metadata_v3(self, provider_search_task_id: str, trace_id: str) -> ProviderTaskMetadataV2:
        api_path = self.SEARCH_TASK_METADATA_FMT_V3.format(provider_search_task_id)
        response = self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))
        return ProviderTaskMetadataV2.parse_obj(response)

    def send_log_event(self, log_event: LogEvent):
        api_path = self.SEND_LOG_EVENT_URI
        try:
            self._with_unauth_retry(
                lambda: self._send_post_req(
                    api_path,
                    trace_id=None,
                    json_data=log_event.dict(exclude_none=True),
                    content_type="application/json",
                    result_format="text",
                    silent=True,
                ),
                need_connection_retry=False,
            )
        except Exception:
            self.send_log_event_unauth(log_event)

    @staticmethod
    def send_log_event_unauth(log_event: LogEvent):
        api_path = _RestClient.SEND_LOG_EVENT_URI
        try:
            requests.post(
                url=urljoin(_RestClient.PROD_BACKEND_URL, api_path),
                json=log_event.dict(exclude_none=True),
                headers=_RestClient._get_base_headers(content_type="application/json"),
            )
        except Exception:
            pass

    # ADS management

    def register_ads(self, request: Dict, trace_id: str) -> str:
        api_path = self.REGISTER_ADS_URI
        response = self._with_unauth_retry(lambda: self._send_post_req(api_path, trace_id, request))
        return response["adsManagementTaskId"]

    def delete_ads(self, ads_definition_id: str, trace_id: str) -> str:
        api_path = self.DELETE_ADS_URI_FMT.format(ads_definition_id)
        response = self._with_unauth_retry(lambda: self._send_delete_req(api_path, trace_id))
        return response["adsManagementTaskId"]

    def activate_datatables(self, request: Dict, trace_id: str):
        api_path = self.ACTIVATE_ADS_URI
        self._with_unauth_retry(lambda: self._send_post_req(api_path, trace_id, request, result_format=None))

    def deactivate_data_tables(self, request: Dict, trace_id: str):
        api_path = self.DEACTIVATE_ADS_URI
        self._with_unauth_retry(lambda: self._send_post_req(api_path, trace_id, request, result_format=None))

    def toggle_ads(self, ads_definition_id: str, trace_id: str):
        api_path = self.TOGGLE_ADS_URI_FMT.format(ads_definition_id)
        return self._with_unauth_retry(lambda: self._send_post_req(api_path, trace_id))

    def poll_ads_management_task_status(self, ads_management_task_id: str, trace_id: str) -> Dict[str, str]:
        api_path = self.POLL_ADS_MANAGEMENT_STATUS_URI_FMT.format(ads_management_task_id)
        response = self._with_unauth_retry(lambda: self._send_get_req(api_path, trace_id))
        return response

    def get_ads_description(self, ads_definition_id: str):
        api_path = self.GET_ADS_DESCRIPTION_URI_FMT.format(ads_definition_id)
        response = self._with_unauth_retry(lambda: self._send_get_req(api_path, None))
        return response

    def get_all_ads_descriptions(self):
        return self._with_unauth_retry(lambda: self._send_get_req(self.GET_ALL_ADS_DESCRIPTIONS_URI, None))

    def get_active_ads_definitions(self):
        return self._with_unauth_retry(lambda: self._send_get_req(self.GET_ACTIVE_ADS_DEFINITIONS_URI, None))

    def get_current_email(self) -> Optional[str]:
        try:
            return jwt.decode(self._get_access_token(), options={"verify_signature": False})["sub"]
        except Exception:
            return None

    # ---

    def _send_get_req(self, api_path: str, trace_id: Optional[str]):
        response = requests.get(
            url=urljoin(self._service_endpoint, api_path), headers=self._get_headers(trace_id=trace_id)
        )

        if response.status_code >= 400:
            raise HttpError(response.text, status_code=response.status_code)

        return response.json()

    def _send_get_file_req(self, api_path: str, trace_id: Optional[str]):
        response = requests.get(
            url=urljoin(self._service_endpoint, api_path), headers=self._get_headers(trace_id=trace_id)
        )

        if response.status_code >= 400:
            raise HttpError(response.text, status_code=response.status_code)

        return response.content

    def _send_post_req(
        self,
        api_path: str,
        trace_id: Optional[str],
        json_data=None,
        data=None,
        content_type=None,
        result_format="json",
        silent=False,
    ):
        response = requests.post(
            url=urljoin(self._service_endpoint, api_path),
            data=data,
            json=json_data,
            headers=self._get_headers(content_type, trace_id=trace_id),
        )

        if response.status_code >= 400:
            if not silent and response.status_code != 401:
                logging.error(f"Failed to execute request to {api_path}: {response}")
            raise HttpError(response.text, status_code=response.status_code)

        if result_format == "json":
            return response.json()
        else:
            return response.text

    def _send_delete_req(
        self,
        api_path: str,
        trace_id: Optional[str],
        content_type=None,
        result_format="json",
        silent=False,
    ):
        response = requests.delete(
            url=urljoin(self._service_endpoint, api_path),
            headers=self._get_headers(content_type, trace_id=trace_id),
        )

        if response.status_code >= 400:
            if not silent and response.status_code != 401:
                logging.error(f"Failed to execute request to {api_path}: {response}")
            raise HttpError(response.text, status_code=response.status_code)

        if result_format == "json":
            return response.json()
        else:
            return response.text

    def _send_post_file_req_v2(
        self,
        api_path,
        files,
        data=None,
        trace_id: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        need_json_response=True,
    ):
        additional_headers = additional_headers or {}
        response = requests.post(
            url=urljoin(self._service_endpoint, api_path),
            data=data,
            files=files,
            headers=self._get_headers(trace_id=trace_id, additional_headers=additional_headers),
        )

        if response.status_code >= 400:
            raise HttpError(response.text, status_code=response.status_code)

        if need_json_response:
            return response.json()
        else:
            return response

    @staticmethod
    def _get_base_headers(
        content_type: Optional[str] = None,
        trace_id: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        additional_headers = additional_headers or {}
        headers = {
            _RestClient.USER_AGENT_HEADER_NAME: _RestClient.USER_AGENT_HEADER_VALUE,
        }
        if content_type:
            headers[_RestClient.CONTENT_TYPE_HEADER_NAME] = content_type
        if trace_id:
            headers[_RestClient.TRACE_ID_HEADER_NAME] = trace_id
        for header_key, header_value in additional_headers.items():
            headers[header_key] = header_value
        return headers

    def _get_headers(
        self,
        content_type: Optional[str] = None,
        trace_id: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ):
        headers = self._get_base_headers(content_type, trace_id, additional_headers or {})
        headers[self.ACCESS_TOKEN_HEADER_NAME] = "Bearer " + self._get_access_token()

        return headers

    @staticmethod
    def _update_columns_request(update_columns):
        return {"columns": [x.to_json() for x in update_columns]}


def _resolve_backend_url(backend_url: Optional[str]) -> str:
    if backend_url is not None:
        return backend_url
    elif UPGINI_URL in os.environ and os.environ[UPGINI_URL]:
        return os.environ[UPGINI_URL]
    else:
        return _RestClient.PROD_BACKEND_URL


def resolve_api_token(api_token: Optional[str]) -> str:
    if api_token is not None:
        return api_token
    elif UPGINI_API_KEY in os.environ and os.environ[UPGINI_API_KEY]:
        return os.environ[UPGINI_API_KEY]
    else:
        return DEMO_API_KEY


def get_rest_client(backend_url: Optional[str] = None, api_token: Optional[str] = None) -> _RestClient:
    url = _resolve_backend_url(backend_url)
    token = resolve_api_token(api_token)

    return _get_rest_client(url, token)


def is_demo_api_key(api_token: Optional[str]) -> bool:
    return api_token is None or api_token == "" or api_token == DEMO_API_KEY


@lru_cache()
def _get_rest_client(backend_url: str, api_token: str) -> _RestClient:
    return _RestClient(backend_url, api_token)


class BackendLogHandler(logging.Handler):
    def __init__(self, rest_client: _RestClient, client_ip: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rest_client = rest_client
        self.track_metrics = None
        self.hostname = "0.0.0.0"
        self.client_ip = client_ip

    def emit(self, record: logging.LogRecord) -> None:
        def task():
            try:
                if self.track_metrics is None or len(self.track_metrics) == 0:
                    self.track_metrics = get_track_metrics(self.client_ip)
                    self.hostname = self.track_metrics.get("ip") or "0.0.0.0"
                text = self.format(record)
                tags = self.track_metrics
                tags["version"] = __version__
                log_event = LogEvent(
                    source="python",
                    tags=",".join([f"{k}:{v}" for k, v in tags.items()]),
                    hostname=self.hostname,
                    message=text,
                    service="PyLib",
                )
                self.rest_client.send_log_event(log_event)
            except Exception:
                pass

        thread = threading.Thread(target=task)
        thread.start()


class LoggerFactory:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                # another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls._instance:
                    cls._instance = super(LoggerFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        super(LoggerFactory).__init__(*args, **kwargs)
        self._loggers: Dict[str, logging.Logger] = {}
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.handlers.clear()

    def get_logger(
        self, backend_url: Optional[str] = None, api_token: Optional[str] = None, client_ip: Optional[str] = None
    ) -> logging.Logger:
        url = _resolve_backend_url(backend_url)
        token = resolve_api_token(api_token)
        key = url + token

        if key in self._loggers:
            return self._loggers[key]

        upgini_logger = logging.getLogger(f"upgini.{hash(key)}")
        upgini_logger.handlers.clear()
        rest_client = get_rest_client(backend_url, api_token)
        datadog_handler = BackendLogHandler(rest_client, client_ip)
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(threadName)s %(name)s %(levelname)s %(message)s",
            timestamp=True,
        )
        datadog_handler.setFormatter(json_formatter)
        upgini_logger.addHandler(datadog_handler)
        self._loggers[key] = upgini_logger

        return upgini_logger


def show_status_error():
    try:
        response = requests.get("https://api.github.com/repos/upgini/upgini/contents/error_status.txt")
        if response.status_code == requests.codes.ok:
            js = response.json()
            content = base64.b64decode(js["content"]).decode("utf-8")
            if len(content) > 0 and not content.isspace():
                print(content)
    except Exception:
        pass
