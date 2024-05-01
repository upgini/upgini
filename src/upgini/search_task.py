import logging
import tempfile
import time
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

from upgini import dataset
from upgini.http import (
    ProviderTaskSummary,
    SearchProgress,
    SearchTaskSummary,
    _RestClient,
    get_rest_client,
    is_demo_api_key,
)
from upgini.metadata import (
    SYSTEM_RECORD_ID,
    FeaturesMetadataV2,
    FileMetadata,
    GeneratedFeatureMetadata,
    ModelTaskType,
    ProviderTaskMetadataV2,
    RuntimeParameters,
)
from upgini.resource_bundle import bundle
from upgini.spinner import Spinner


class SearchTask:
    summary: Optional[SearchTaskSummary]
    POLLING_DELAY_SECONDS = 5
    PROTECT_FROM_RATE_LIMIT = True

    def __init__(
        self,
        search_task_id: str,
        dataset: Optional["dataset.Dataset"] = None,
        return_scores: bool = False,
        extract_features: bool = False,
        accurate_model: bool = False,
        initial_search_task_id: Optional[str] = None,
        task_type: Optional[ModelTaskType] = None,
        rest_client: Optional[_RestClient] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.search_task_id = search_task_id
        self.initial_search_task_id = initial_search_task_id
        self.dataset = dataset
        self.return_scores = return_scores
        self.extract_features = extract_features
        self.accurate_model = accurate_model
        self.task_type = task_type
        self.summary = None
        self.rest_client = rest_client
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("muted_logger")
            self.logger.setLevel("FATAL")
        self.provider_metadata_v2: Optional[List[ProviderTaskMetadataV2]] = None
        self.unused_features_for_generation: Optional[List[str]] = None

    def get_progress(self, trace_id: str) -> SearchProgress:
        return self.rest_client.get_search_progress(trace_id, self.search_task_id)

    def poll_result(self, trace_id: str, quiet: bool = False, check_fit: bool = False) -> "SearchTask":
        completed_statuses = {"COMPLETED", "VALIDATION_COMPLETED"}
        failed_statuses = {"FAILED", "VALIDATION_FAILED", "EMPTY_INTERSECTION"}
        submitted_statuses = {"SUBMITTED", "VALIDATION_SUBMITTED"}
        if not quiet:
            print(bundle.get("polling_search_task").format(self.search_task_id))
            if is_demo_api_key(self.rest_client._refresh_token):
                print(bundle.get("polling_unregister_information"))
        search_task_id = self.initial_search_task_id if self.initial_search_task_id is not None else self.search_task_id

        try:
            with Spinner():
                if self.PROTECT_FROM_RATE_LIMIT:
                    time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
                self.summary = self.rest_client.search_task_summary_v2(trace_id, search_task_id)
                while self.summary.status not in completed_statuses and (
                    not check_fit or "VALIDATION" not in self.summary.status
                ):
                    time.sleep(self.POLLING_DELAY_SECONDS)
                    self.summary = self.rest_client.search_task_summary_v2(trace_id, search_task_id)
                    if self.summary.status in failed_statuses:
                        self.logger.error(f"Search {search_task_id} failed with status {self.summary.status}")
                        raise RuntimeError(bundle.get("search_task_failed_status"))
                    if (
                        self.summary.status in submitted_statuses
                        and len(self._get_provider_summaries(self.summary)) == 0
                    ):
                        self.logger.error(f"No provider summaries for search {search_task_id}")
                        raise RuntimeError(bundle.get("no_one_provider_respond"))
                    time.sleep(self.POLLING_DELAY_SECONDS)
        except KeyboardInterrupt as e:
            if not check_fit:
                print(bundle.get("search_stopping"))
                self.rest_client.stop_search_task_v2(trace_id, search_task_id)
                self.logger.warning(f"Search {search_task_id} stopped by user")
                print(bundle.get("search_stopped"))
            raise e
        print()

        has_completed_provider_task = False
        for provider_summary in self._get_provider_summaries(self.summary):
            if provider_summary.status == "COMPLETED":
                has_completed_provider_task = True

        if not has_completed_provider_task and not check_fit:
            error_messages = [self._error_message(x) for x in self._get_provider_summaries(self.summary)]
            if len(error_messages) == 1 and (error_messages[0] is None or error_messages[0].endswith("Internal error")):
                self.logger.error(f"Search failed with error: {error_messages[0]}")
                raise RuntimeError(bundle.get("all_providers_failed"))
            else:
                self.logger.error(f"Search failed with errors: {','.join(error_messages)}")
                raise RuntimeError(bundle.get("all_providers_failed_with_error").format(",".join(error_messages)))

        if self.summary.status in ["COMPLETED", "VALIDATION_COMPLETED"] or (
            check_fit and "VALIDATION" in self.summary.status
        ):
            self.provider_metadata_v2 = []
            self.unused_features_for_generation = []
            for provider_summary in self.summary.initial_important_providers:
                if provider_summary.status == "COMPLETED":
                    self.provider_metadata_v2.append(
                        self.rest_client.get_provider_search_metadata_v3(provider_summary.ads_search_task_id, trace_id)
                    )
                    if provider_summary.unused_features_for_generation is not None:
                        self.unused_features_for_generation.extend(provider_summary.unused_features_for_generation)

        return self

    def get_all_features_metadata_v2(self) -> Optional[List[FeaturesMetadataV2]]:
        if self.provider_metadata_v2 is None:
            return None

        features_meta = []
        for meta in self.provider_metadata_v2:
            features_meta.extend(meta.features)

        return features_meta

    def get_zero_hit_rate_search_keys(self) -> Optional[List[str]]:
        if self.provider_metadata_v2 is None:
            return None

        zero_hit_search_keys = set()
        for meta in self.provider_metadata_v2:
            if meta.zero_hit_rate_search_keys is not None:
                zero_hit_search_keys.update(meta.zero_hit_rate_search_keys)

        return list(zero_hit_search_keys)

    def get_features_for_transform(self) -> Optional[List[str]]:
        if self.provider_metadata_v2 is None:
            return None

        features_for_transform = set()
        for meta in self.provider_metadata_v2:
            if meta.features_used_for_embeddings is not None:
                features_for_transform.update(meta.features_used_for_embeddings)

        return list(features_for_transform)

    def get_shuffle_kfold(self) -> Optional[bool]:
        if self.provider_metadata_v2 is None:
            return None

        for meta in self.provider_metadata_v2:
            if meta.shuffle_kfold is not None:
                return meta.shuffle_kfold

    def get_autofe_metadata(self) -> Optional[List[GeneratedFeatureMetadata]]:
        if self.provider_metadata_v2 is None:
            return None

        for meta in self.provider_metadata_v2:
            if meta.generated_features is not None:
                return meta.generated_features

    @staticmethod
    def _get_provider_summaries(summary: SearchTaskSummary) -> List[ProviderTaskSummary]:
        if summary.status in {
            "VALIDATION_CREATED",
            "VALIDATION_SUBMITTED",
            "VALIDATION_COMPLETED",
            "VALIDATION_FAILED",
        }:
            return summary.validation_important_providers
        else:
            return summary.initial_important_providers

    @staticmethod
    def _error_message(provider_summary: ProviderTaskSummary):
        if provider_summary.error_message:
            return provider_summary.error_message
        else:
            if provider_summary.status == "TIMED_OUT":
                return bundle.get("search_timed_out")
            elif provider_summary.status == "EMPTY_INTERSECTION":
                return "Empty intersection"
            else:
                return bundle.get("search_other_error")

    def validation(
        self,
        trace_id: str,
        validation_dataset: "dataset.Dataset",
        start_time: int,
        extract_features: bool = False,
        runtime_parameters: Optional[RuntimeParameters] = None,
        exclude_features_sources: Optional[List[str]] = None,
        metrics_calculation: bool = False,
        silent_mode: bool = False,
        progress_bar=None,
        progress_callback=None,
    ) -> "SearchTask":
        return validation_dataset.validation(
            trace_id,
            self.search_task_id,
            start_time=start_time,
            return_scores=True,
            extract_features=extract_features,
            runtime_parameters=runtime_parameters,
            exclude_features_sources=exclude_features_sources,
            metrics_calculation=metrics_calculation,
            silent_mode=silent_mode,
            progress_bar=progress_bar,
            progress_callback=progress_callback,
        )

    def _check_finished_initial_search(self):
        if self.provider_metadata_v2 is None or len(self.provider_metadata_v2) == 0:
            raise RuntimeError(bundle.get("search_not_started"))

    def _check_finished_validation_search(self) -> List[ProviderTaskSummary]:
        if self.summary is None or len(self.summary.validation_important_providers) == 0:
            raise RuntimeError(f"Validation search didn't start. summary: {self.summary}")
        return self.summary.validation_important_providers

    def initial_max_hit_rate_v2(self) -> Optional[float]:
        if self.provider_metadata_v2 is not None:
            return max([meta.hit_rate_metrics.hit_rate_percent for meta in self.provider_metadata_v2])

    def get_all_initial_raw_features(self, trace_id: str, metrics_calculation: bool = False) -> Optional[pd.DataFrame]:
        self._check_finished_initial_search()
        if self.PROTECT_FROM_RATE_LIMIT:
            time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
        return _get_all_initial_raw_features_cached(
            self.rest_client._service_endpoint,
            self.rest_client._refresh_token,
            trace_id,
            self.search_task_id,
            metrics_calculation,
            self.PROTECT_FROM_RATE_LIMIT,
        )

    def get_target_outliers(self, trace_id: str) -> Optional[pd.DataFrame]:
        self._check_finished_initial_search()
        return _get_target_outliers_cached(
            self.rest_client._service_endpoint,
            self.rest_client._refresh_token,
            trace_id,
            self.search_task_id,
            self.PROTECT_FROM_RATE_LIMIT,
        )

    def get_max_initial_eval_set_hit_rate_v2(self) -> Optional[Dict[int, float]]:
        if self.provider_metadata_v2 is not None:
            hit_rate_dict = {}
            for provider_meta in self.provider_metadata_v2:
                if provider_meta.eval_set_metrics is not None:
                    for eval_metrics in provider_meta.eval_set_metrics:
                        eval_idx = eval_metrics.eval_set_index
                        new_hit_rate = eval_metrics.hit_rate_metrics.hit_rate_percent
                        if eval_idx not in hit_rate_dict.keys() or new_hit_rate > hit_rate_dict[eval_idx]:
                            hit_rate_dict[eval_idx] = new_hit_rate
            return hit_rate_dict

    def get_all_validation_raw_features(self, trace_id: str, metrics_calculation=False) -> Optional[pd.DataFrame]:
        self._check_finished_validation_search()
        return _get_all_validation_raw_features_cached(
            self.rest_client._service_endpoint,
            self.rest_client._refresh_token,
            trace_id,
            self.search_task_id,
            metrics_calculation,
            self.PROTECT_FROM_RATE_LIMIT,
        )

    def get_file_metadata(self, trace_id: str) -> FileMetadata:
        return self.rest_client.get_search_file_metadata(self.search_task_id, trace_id)


@lru_cache
def _get_all_initial_raw_features_cached(
    endpoint: Optional[str],
    api_key: Optional[str],
    trace_id: str,
    search_task_id: str,
    metrics_calculation: bool,
    protect_from_rate_limit: bool = True,
) -> Optional[pd.DataFrame]:
    if protect_from_rate_limit:
        time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
    features_response = get_rest_client(endpoint, api_key).get_search_features_v2(
        trace_id, search_task_id, metrics_calculation
    )
    result_df = None
    for feature_block in features_response["adsSearchTaskFeaturesDTO"]:
        if feature_block["searchType"] == "INITIAL":
            features_id = feature_block["adsSearchTaskFeaturesId"]
            if protect_from_rate_limit:
                time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
            features_df = _download_features_file(endpoint, api_key, trace_id, features_id, metrics_calculation)
            if result_df is None:
                result_df = features_df
            else:
                result_df = pd.merge(result_df, features_df, how="outer", on=SYSTEM_RECORD_ID)

    if result_df is not None:
        for column in result_df.columns:
            if column.startswith("etalon_"):
                result_df.rename(columns={column: column[7:]}, inplace=True)
    return result_df


@lru_cache
def _get_all_validation_raw_features_cached(
    endpoint: Optional[str],
    api_key: Optional[str],
    trace_id: str,
    search_task_id: str,
    metrics_calculation=False,
    protect_from_rate_limit: bool = True,
) -> Optional[pd.DataFrame]:
    if protect_from_rate_limit:
        time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
    features_response = get_rest_client(endpoint, api_key).get_search_features_v2(
        trace_id, search_task_id, metrics_calculation
    )
    result_df = None
    for feature_block in features_response["adsSearchTaskFeaturesDTO"]:
        if feature_block["searchType"] == "VALIDATION":
            features_id = feature_block["adsSearchTaskFeaturesId"]
            if protect_from_rate_limit:
                time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
            features_df = _download_features_file(endpoint, api_key, trace_id, features_id, metrics_calculation)
            if result_df is None:
                result_df = features_df
            else:
                result_df = pd.merge(result_df, features_df, how="outer", on=SYSTEM_RECORD_ID)

    return result_df


@lru_cache
def _get_target_outliers_cached(
    endpoint: Optional[str],
    api_key: Optional[str],
    trace_id: str,
    search_task_id: str,
    protect_from_rate_limit: bool = True,
) -> pd.DataFrame:
    if protect_from_rate_limit:
        time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
    target_outliers_response = get_rest_client(endpoint, api_key).get_search_target_outliners(trace_id, search_task_id)
    result_df = None
    for dto in target_outliers_response["adsSearchTaskTargetOutliersDTO"]:
        target_outliers_id = dto["adsSearchTaskTargetOutliersId"]
        if protect_from_rate_limit:
            time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
        file_content = get_rest_client(endpoint, api_key).get_search_target_outliners_file(trace_id, target_outliers_id)
        target_outliers_df = _read_parquet(file_content, "target_outliers.parquet")
        if result_df is None:
            result_df = target_outliers_df
        else:
            # TODO another strategy of merge
            pass
            # result_df = pd.merge(result_df, target_outliers_df, how="outer", on=SYSTEM_RECORD_ID)

    return result_df


def _download_features_file(
    endpoint: Optional[str], api_key: Optional[str], trace_id: str, features_id: str, metrics_calculation: bool
) -> pd.DataFrame:
    file_content = get_rest_client(endpoint, api_key).get_search_features_file_v2(
        trace_id, features_id, metrics_calculation
    )
    return _read_parquet(file_content)


def _read_parquet(file_content: bytes, file_name: str = "features.parquet"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_name = f"{tmp_dir}/{file_name}"
        with open(tmp_file_name, "wb") as gzip_file:
            gzip_file.write(file_content)
        return pd.read_parquet(tmp_file_name, engine="fastparquet")
