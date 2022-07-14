import os
import tempfile
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from upgini import dataset
from upgini.http import LoggerFactory, ProviderTaskSummary, SearchTaskSummary, get_rest_client
from upgini.metadata import (
    SYSTEM_RECORD_ID,
    FileMetadata,
    ModelTaskType,
    RuntimeParameters,
)
from upgini.spinner import Spinner


class SearchTask:
    summary: Optional[SearchTaskSummary]

    def __init__(
        self,
        search_task_id: str,
        dataset: Optional["dataset.Dataset"] = None,
        return_scores: bool = False,
        extract_features: bool = False,
        accurate_model: bool = False,
        initial_search_task_id: Optional[str] = None,
        task_type: Optional[ModelTaskType] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.search_task_id = search_task_id
        self.initial_search_task_id = initial_search_task_id
        self.dataset = dataset
        self.return_scores = return_scores
        self.extract_features = extract_features
        self.accurate_model = accurate_model
        self.task_type = task_type
        self.summary = None
        self.endpoint = endpoint
        self.api_key = api_key
        self.logger = LoggerFactory().get_logger(endpoint, api_key)

    def poll_result(self, trace_id: str, quiet: bool = False) -> "SearchTask":
        completed_statuses = {"COMPLETED", "VALIDATION_COMPLETED"}
        failed_statuses = {"FAILED", "VALIDATION_FAILED"}
        submitted_statuses = {"SUBMITTED", "VALIDATION_SUBMITTED"}
        if not quiet:
            print(
                f"Running search request with search_id={self.search_task_id}\n"
                "We'll send email notification once it's completed, "
                "just use your personal api_key from profile.upgini.com"
            )
        search_task_id = self.initial_search_task_id if self.initial_search_task_id is not None else self.search_task_id

        try:
            with Spinner():
                time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
                self.summary = get_rest_client(self.endpoint, self.api_key).search_task_summary_v2(
                    trace_id, search_task_id
                )
                while self.summary.status not in completed_statuses:
                    time.sleep(5)
                    self.summary = get_rest_client(self.endpoint, self.api_key).search_task_summary_v2(
                        trace_id, search_task_id
                    )
                    if self.summary.status in failed_statuses:
                        self.logger.error(f"Search {search_task_id} failed with status {self.summary.status}")
                        raise RuntimeError("Oh! Server did something wrong, please retry with new search request")
                    if (
                        self.summary.status in submitted_statuses
                        and len(self._get_provider_summaries(self.summary)) == 0
                    ):
                        self.logger.error(f"No provider summaries for search {search_task_id}")
                        raise RuntimeError(
                            "No datasets found to intersect"
                            "Try with another set of search keys or different time period"
                        )
                    time.sleep(5)
        except KeyboardInterrupt as e:
            print("Search interrupted. Stopping search request")
            get_rest_client(self.endpoint, self.api_key).stop_search_task_v2(trace_id, search_task_id)
            self.logger.warn(f"Search {search_task_id} stopped by user")
            print("Search request stopped")
            raise e
        print()

        has_completed_provider_task = False
        for provider_summary in self._get_provider_summaries(self.summary):
            if provider_summary.status == "COMPLETED":
                has_completed_provider_task = True

        if not has_completed_provider_task:
            error_messages = [self._error_message(x) for x in self._get_provider_summaries(self.summary)]
            if len(error_messages) == 1 and (error_messages[0] is None or error_messages[0].endswith("Internal error")):
                self.logger.error(f"Search failed with error: {error_messages[0]}")
                raise RuntimeError("All search tasks in the request have failed")
            else:
                self.logger.error(f"Search failed with errors: {','.join(error_messages)}")
                raise RuntimeError("All search tasks in the request have failed: " + ",".join(error_messages) + ".")

        return self

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
                return "Search request timed out"
            elif provider_summary.status == "EMPTY_INTERSECTION":
                return "Datasets doesn't intersect with uploaded file"
            else:
                return "Internal error"

    def validation(
        self,
        trace_id: str,
        validation_dataset: "dataset.Dataset",
        extract_features: bool = False,
        runtime_parameters: Optional[RuntimeParameters] = None,
        silent_mode: bool = False,
    ) -> "SearchTask":
        return validation_dataset.validation(
            trace_id,
            self.search_task_id,
            return_scores=True,
            extract_features=extract_features,
            runtime_parameters=runtime_parameters,
            silent_mode=silent_mode,
        )

    def _check_finished_initial_search(self) -> List[ProviderTaskSummary]:
        if self.summary is None or len(self.summary.initial_important_providers) == 0:
            raise RuntimeError("Initial search didn't start.")
        return self.summary.initial_important_providers

    def _check_finished_validation_search(self) -> List[ProviderTaskSummary]:
        if self.summary is None or len(self.summary.validation_important_providers) == 0:
            raise RuntimeError(f"Validation search didn't start. summary: {self.summary}")
        return self.summary.validation_important_providers

    @staticmethod
    def _has_metric(provider_summaries: List[ProviderTaskSummary], metric_code: str) -> bool:
        for provider_summary in provider_summaries:
            for code in provider_summary.metrics.keys():
                if code == metric_code:
                    return True

        return False

    @staticmethod
    def _metric_by_provider(provider_summaries: List[ProviderTaskSummary], metric_code: str) -> List[Dict[str, str]]:
        metric_by_provider = []
        for provider_summary in provider_summaries:
            for code, value in provider_summary.metrics.items():
                if code == metric_code:
                    metric_by_provider.append(
                        {
                            "provider_id": provider_summary.provider_id,
                            "value": value,
                        }
                    )
        return metric_by_provider

    @staticmethod
    def _ads_search_task_id_by_provider_id(provider_summaries: List[ProviderTaskSummary], provider_id: str) -> str:
        for provider_summary in provider_summaries:
            if provider_summary.provider_id == provider_id:
                return provider_summary.ads_search_task_id
        raise RuntimeError(f"Provider {provider_id} not found.")

    @staticmethod
    def _search_task_id_by_provider_id(provider_summaries: List[ProviderTaskSummary], provider_id: str) -> str:
        for provider_summary in provider_summaries:
            if provider_summary.provider_id == provider_id:
                return provider_summary.search_task_id
        raise RuntimeError(f"Provider {provider_id} not found.")

    @staticmethod
    def _model_id_by_provider(provider_summaries: List[ProviderTaskSummary]) -> pd.DataFrame:
        result = []
        for provider_summary in provider_summaries:
            result.append(
                {
                    "provider_id": provider_summary.provider_id,
                    "model_id": provider_summary.ads_search_task_id,
                }
            )
        return pd.DataFrame(result)

    @staticmethod
    def _max_by_metric(provider_summaries: List[ProviderTaskSummary], metric_code: str) -> Dict[str, Any]:
        max_provider = None
        max_metric = None
        for x in SearchTask._metric_by_provider(provider_summaries, metric_code):
            current_metric = float(x["value"])
            if max_metric is None or current_metric > max_metric:
                max_provider = x["provider_id"]
                max_metric = current_metric

        if max_metric is None:
            raise RuntimeError(f"There is no {metric_code} available for search task.")
        else:
            return {"provider_id": max_provider, "value": max_metric}

    def initial_max_auc(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "AUC"):
            return self._max_by_metric(provider_summaries, "AUC")
        else:
            return None

    def initial_max_accuracy(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "ACCURACY"):
            return self._max_by_metric(provider_summaries, "ACCURACY")
        else:
            return None

    def initial_max_rmse(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "RMSE"):
            return self._max_by_metric(provider_summaries, "RMSE")
        else:
            return None

    def initial_max_uplift(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "UPLIFT"):
            return self._max_by_metric(provider_summaries, "UPLIFT")
        else:
            return None

    def initial_max_hit_rate(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "HIT_RATE"):
            return self._max_by_metric(provider_summaries, "HIT_RATE")
        else:
            return None

    def _initial_min_hit_rate(self) -> float:
        provider_summaries = self._check_finished_initial_search()
        min_hit_rate = None
        for x in self._metric_by_provider(provider_summaries, "HIT_RATE"):
            current_value = float(x["value"])
            if min_hit_rate is None or current_value < min_hit_rate:
                min_hit_rate = current_value

        if min_hit_rate is None:
            raise RuntimeError("There is no hit rate available for search task.")
        else:
            return min_hit_rate

    def initial_gini(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "GINI"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "GINI")).rename(
                columns={"value": "gini"}, inplace=False
            )
        else:
            return None

    def initial_auc(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "AUC"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "AUC")).rename(
                columns={"value": "roc-auc"}, inplace=False
            )
        else:
            return None

    def initial_accuracy(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "ACCURACY"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "ACCURACY")).rename(
                columns={"value": "accuracy"}, inplace=False
            )
        else:
            return None

    def initial_rmse(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "RMSE"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "RMSE")).rename(
                columns={"value": "rmse"}, inplace=False
            )
        else:
            return None

    def initial_uplift(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "UPLIFT"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "UPLIFT")).rename(
                columns={"value": "uplift"}, inplace=False
            )
        else:
            return None

    def initial_hit_rate(self) -> pd.DataFrame:
        provider_summaries = self._check_finished_initial_search()
        result = pd.DataFrame(self._metric_by_provider(provider_summaries, "HIT_RATE"))
        result.rename(
            columns={"value": "hit_rate"}, inplace=True
        )
        return result

    def initial_metadata(self) -> pd.DataFrame:
        provider_summaries = self._check_finished_initial_search()
        quality_df = None
        auc_df = self.initial_auc()
        gini_df = self.initial_gini()
        accuracy_df = self.initial_accuracy()
        rmse_df = self.initial_rmse()
        if auc_df is not None:
            quality_df = auc_df
        elif gini_df is not None:
            quality_df = gini_df
        elif accuracy_df is not None:
            quality_df = accuracy_df
        elif rmse_df is not None:
            quality_df = rmse_df
        uplift_df = self.initial_uplift()
        hit_rate_df = self.initial_hit_rate()
        model_id_df = self._model_id_by_provider(provider_summaries)
        result = pd.merge(model_id_df, hit_rate_df, on="provider_id")
        if quality_df is not None:
            result = pd.merge(result, quality_df, on="provider_id")
        if uplift_df is not None:
            result = pd.merge(result, uplift_df, on="provider_id")
        return result

    def get_initial_scores_by_provider_id(self, trace_id: str, provider_id: str) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_initial_search()
        scores_response = get_rest_client(self.endpoint, self.api_key).get_search_scores_v2(
            trace_id, self.search_task_id
        )
        ads_search_task_id = self._ads_search_task_id_by_provider_id(provider_summaries, provider_id)
        scores_id = None
        for score_block in scores_response["adsSearchTaskTrainedScoresDTO"]:
            if score_block["adsSearchTaskId"] == ads_search_task_id:
                if score_block["trainedModelScoresType"] == "INITIAL_ETALON_AND_ADS":
                    scores_id = score_block["adsSearchTaskScoresId"]
                elif score_block["trainedModelScoresType"] == "INITIAL_ADS" and not scores_id:
                    scores_id = score_block["adsSearchTaskScoresId"]

        if scores_id is None:
            self.logger.error(f"Initial scores by provider {provider_id} not found")
            print(f"Provider {provider_id} task wasn't completed in initial search")
            return None

        gzip_file_content = get_rest_client(self.endpoint, self.api_key).get_search_scores_file_v2(trace_id, scores_id)
        with tempfile.TemporaryDirectory() as tmp_dir:
            gzip_file_name = "{0}/scores.gzip".format(tmp_dir)
            with open(gzip_file_name, "wb") as gzip_file:
                gzip_file.write(gzip_file_content)
            scores = pd.read_csv(gzip_file_name, compression="gzip", low_memory=False)
            # TODO implement client hashing
            # if self.initial_dataset.initial_to_hashed is not None:
            # # Hardcode with etalon msisdn - use system_id
            #     scores = pd.merge(scores, self.initial_dataset.initial_to_hashed, \
            #        on=["etalon_msisdn", "phone_hashed"])
            #     scores["etalon_msisdn"] = scores[self.initial_dataset.metadata.phone_column]
            #     scores.drop(columns="phone_hashed", inplace=True)
            # if self.initial_dataset.drop_phone_column:
            #     scores.drop(columns="etalon_" + self.initial_dataset.metadata.phone_column, inplace=True)
            # if self.initial_dataset.drop_date_column:
            #     scores.drop(columns="etalon_" + self.initial_dataset.metadata.date_column, inplace=True)
            return scores  # type: ignore

    def _download_features_file(self, trace_id: str, features_id: str) -> pd.DataFrame:
        time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
        gzip_file_content = get_rest_client(self.endpoint, self.api_key).get_search_features_file_v2(
            trace_id, features_id
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            gzip_file_name = "{0}/features.parquet".format(tmp_dir)
            with open(gzip_file_name, "wb") as gzip_file:
                gzip_file.write(gzip_file_content)
            return pd.read_parquet(gzip_file_name)

    def get_initial_raw_features_by_provider_id(self, trace_id: str, provider_id: str) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_initial_search()
        time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
        features_response = get_rest_client(self.endpoint, self.api_key).get_search_features_v2(
            trace_id, self.search_task_id
        )
        ads_search_task_id = self._ads_search_task_id_by_provider_id(provider_summaries, provider_id)
        features_id = None
        for feature_block in features_response["adsSearchTaskFeaturesDTO"]:
            if feature_block["adsSearchTaskId"] == ads_search_task_id and feature_block["searchType"] == "INITIAL":
                features_id = feature_block["adsSearchTaskFeaturesId"]

        if features_id is None:
            self.logger.error(f"Initial features by provider {provider_id} not found")
            print(f"Provider {provider_id} task wasn't completed in initial search")
            return None

        return self._download_features_file(trace_id, features_id)

    def get_all_initial_raw_features(self, trace_id: str) -> Optional[pd.DataFrame]:
        self._check_finished_initial_search()
        return self._get_all_initial_raw_features(trace_id, self.search_task_id)

    @lru_cache()
    def _get_all_initial_raw_features(self, trace_id: str, search_task_id: str) -> Optional[pd.DataFrame]:
        time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
        features_response = get_rest_client(self.endpoint, self.api_key).get_search_features_v2(
            trace_id, search_task_id
        )
        result_df = None
        for feature_block in features_response["adsSearchTaskFeaturesDTO"]:
            if feature_block["searchType"] == "INITIAL":
                features_id = feature_block["adsSearchTaskFeaturesId"]
                features_df = self._download_features_file(trace_id, features_id)
                if result_df is None:
                    result_df = features_df
                else:
                    result_df = pd.merge(result_df, features_df, how="outer", on=SYSTEM_RECORD_ID)

        if result_df is not None:
            for column in result_df.columns:
                if column.startswith("etalon_"):
                    result_df.rename(columns={column: column[7:]}, inplace=True)
        return result_df

    def download_model_by_provider_id(self, trace_id: str, provider_id: str, model_path: str) -> None:
        provider_summaries = self._check_finished_initial_search()
        models_response = get_rest_client(self.endpoint, self.api_key).get_search_models_v2(
            trace_id, self.search_task_id
        )
        ads_search_task_id = self._ads_search_task_id_by_provider_id(provider_summaries, provider_id)
        model_id = None
        for model_block in models_response["adsSearchTaskTrainedModelDTO"]:
            if model_block["adsSearchTaskId"] == ads_search_task_id:
                if model_block["trainedModelType"] == "ETALON_AND_ADS":
                    model_id = model_block["adsSearchTaskTrainedModelId"]
                elif model_block["trainedModelType"] == "ADS" and model_id is None:
                    model_id = model_block["adsSearchTaskTrainedModelId"]

        if model_id is None:
            self.logger.error(f"Model by provider {provider_id} not found")
            print(f"Provider's {provider_id} task wasn't completed in initial search")
            return None

        model_bytes = get_rest_client(self.endpoint, self.api_key).get_search_model_file_v2(trace_id, model_id)
        if model_path.startswith("/") and not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        with open(model_path, "wb") as model_file:
            model_file.write(model_bytes)
        print(f"Model successfully saved to {model_path}")

    def get_max_initial_eval_set_metrics(self) -> Optional[List[dict]]:
        provider_summaries = self._check_finished_initial_search()
        max_idx = None
        max_hit_rate = None
        for idx, summary in enumerate(provider_summaries):
            if summary.eval_set_metrics is not None:
                for eval in summary.eval_set_metrics:
                    if max_idx is None:
                        max_idx = idx
                    if max_hit_rate is None:
                        max_hit_rate = eval.hit_rate
                    elif eval.hit_rate > max_hit_rate:
                        max_hit_rate = eval.hit_rate
                        max_idx = idx

        if max_idx is not None:
            eval_set_metrics = provider_summaries[max_idx].eval_set_metrics
            if eval_set_metrics is not None:
                return [eval.dict(exclude_none=True) for eval in eval_set_metrics]

        return None

    def validation_max_auc(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_validation_search()
        if self._has_metric(provider_summaries, "AUC"):
            return self._max_by_metric(provider_summaries, "AUC")
        else:
            return None

    def validation_max_accuracy(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "ACCURACY"):
            return self._max_by_metric(provider_summaries, "ACCURACY")
        else:
            return None

    def validation_max_rmse(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_initial_search()
        if self._has_metric(provider_summaries, "RMSE"):
            return self._max_by_metric(provider_summaries, "RMSE")
        else:
            return None

    def validation_max_uplift(self) -> Optional[Dict[str, Any]]:
        provider_summaries = self._check_finished_validation_search()
        if self._has_metric(provider_summaries, "UPLIFT"):
            return self._max_by_metric(provider_summaries, "UPLIFT")
        else:
            return None

    def validation_gini(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_validation_search()
        if self._has_metric(provider_summaries, "GINI"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "GINI")).rename(
                columns={"value": "gini"}, inplace=False
            )
        else:
            return None

    def validation_auc(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_validation_search()
        if self._has_metric(provider_summaries, "AUC"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "AUC")).rename(
                columns={"value": "roc-auc"}, inplace=False
            )
        else:
            return None

    def validation_accuracy(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_validation_search()
        if self._has_metric(provider_summaries, "ACCURACY"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "ACCURACY")).rename(
                columns={"value": "accuracy"}, inplace=False
            )
        else:
            return None

    def validation_rmse(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_validation_search()
        if self._has_metric(provider_summaries, "RMSE"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "RMSE")).rename(
                columns={"value": "rmse"}, inplace=False
            )
        else:
            return None

    def validation_uplift(self) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_validation_search()
        if self._has_metric(provider_summaries, "UPLIFT"):
            return pd.DataFrame(self._metric_by_provider(provider_summaries, "UPLIFT")).rename(
                columns={"value": "uplift"}, inplace=False
            )
        else:
            return None

    def validation_hit_rate(self) -> pd.DataFrame:
        provider_summaries = self._check_finished_validation_search()
        result = pd.DataFrame(self._metric_by_provider(provider_summaries, "HIT_RATE"))
        result.rename(
            columns={"value": "hit_rate"}, inplace=True
        )
        return result

    def _validation_min_hit_rate(self) -> float:
        provider_summaries = self._check_finished_validation_search()
        min_hit_rate = None
        for x in self._metric_by_provider(provider_summaries, "HIT_RATE"):
            current_value = float(x["value"])
            if min_hit_rate is None or current_value < min_hit_rate:
                min_hit_rate = current_value

        if min_hit_rate is None:
            raise RuntimeError("There is no hit rate available for search task")
        else:
            return min_hit_rate

    def validation_metadata(self) -> pd.DataFrame:
        provider_summaries = self._check_finished_validation_search()
        quality_df = None
        gini_df = self.validation_gini()
        auc_df = self.validation_auc()
        accuracy_df = self.validation_accuracy()
        rmse_df = self.validation_rmse()
        if auc_df is not None:
            quality_df = auc_df
        elif gini_df is not None:
            quality_df = gini_df
        elif accuracy_df is not None:
            quality_df = accuracy_df
        elif rmse_df is not None:
            quality_df = rmse_df
        uplift_df = self.validation_uplift()
        hit_rate_df = self.validation_hit_rate()
        model_id_df = self._model_id_by_provider(provider_summaries)
        result = pd.merge(model_id_df, hit_rate_df, on="provider_id")
        if quality_df is not None:
            result = pd.merge(result, quality_df, on="provider_id")
        if uplift_df is not None:
            result = pd.merge(result, uplift_df, on="provider_id")
        return result

    def get_validation_scores_by_provider_id(self, trace_id: str, provider_id: str) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_validation_search()
        validation_task_id = self._search_task_id_by_provider_id(provider_summaries, provider_id)
        scores_response = get_rest_client(self.endpoint, self.api_key).get_search_scores_v2(
            trace_id, validation_task_id
        )
        ads_search_task_id = self._ads_search_task_id_by_provider_id(provider_summaries, provider_id)
        scores_id = None
        for score_block in scores_response["adsSearchTaskTrainedScoresDTO"]:
            if score_block["adsSearchTaskId"] == ads_search_task_id:
                if score_block["trainedModelScoresType"] == "VALIDATION_ETALON_AND_ADS":
                    scores_id = score_block["adsSearchTaskScoresId"]
                elif score_block["trainedModelScoresType"] == "VALIDATION_ADS" and not scores_id:
                    scores_id = score_block["adsSearchTaskScoresId"]

        if scores_id is None:
            self.logger.error(f"Validation scores by provider {provider_id} not found")
            print("Provider ", provider_id, " not found in validation search")
            return None

        gzip_file_content = get_rest_client(self.endpoint, self.api_key).get_search_scores_file_v2(trace_id, scores_id)
        with tempfile.TemporaryDirectory() as tmp_dir:
            gzip_file_name = "{0}/scores.gzip".format(tmp_dir)
            with open(gzip_file_name, "wb") as gzip_file:
                gzip_file.write(gzip_file_content)
            scores = pd.read_csv(gzip_file_name, compression="gzip", low_memory=False)
            # TODO support client hashing
            # if self.validation_dataset.initial_to_hashed is not None:
            #     scores = pd.merge(scores, self.validation_dataset.initial_to_hashed, \
            # on=["etalon_msisdn", "phone_hashed"]) # Hardcode with etalon msisdn - use system_id
            #     scores["etalon_msisdn"] = scores[self.validation_dataset.metadata.phone_column]
            #     scores.drop(columns="phone_hashed", inplace=True)
            # if self.validation_dataset.drop_phone_column:
            #     scores.drop(columns="etalon_" + self.validation_dataset.metadata.phone_column, inplace=True)
            # if self.validation_dataset.drop_date_column:
            #     scores.drop(columns="etalon_" + self.validation_dataset.metadata.date_column, inplace=True)
            return scores  # type: ignore

    def get_validation_raw_features_by_provider_id(self, trace_id: str, provider_id: str) -> Optional[pd.DataFrame]:
        provider_summaries = self._check_finished_validation_search()
        validation_task_id = self._search_task_id_by_provider_id(provider_summaries, provider_id)
        time.sleep(1)
        features_response = get_rest_client(self.endpoint, self.api_key).get_search_features_v2(
            trace_id, validation_task_id
        )
        ads_search_task_id = self._ads_search_task_id_by_provider_id(provider_summaries, provider_id)
        features_id = None
        for feature_block in features_response["adsSearchTaskFeaturesDTO"]:
            if feature_block["adsSearchTaskId"] == ads_search_task_id and feature_block["searchType"] == "VALIDATION":
                features_id = feature_block["adsSearchTaskFeaturesId"]

        if features_id is None:
            self.logger.error(f"Validation features by provider {provider_id} not found")
            print(f"Features for provider {provider_id} not found in validation search")
            return None

        time.sleep(1)
        gzip_file_content = get_rest_client(self.endpoint, self.api_key).get_search_features_file_v2(
            trace_id, features_id
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            gzip_file_name = "{0}/features.parquet".format(tmp_dir)
            with open(gzip_file_name, "wb") as gzip_file:
                gzip_file.write(gzip_file_content)
            return pd.read_parquet(gzip_file_name)

    def get_all_validation_raw_features(self, trace_id: str) -> Optional[pd.DataFrame]:
        self._check_finished_validation_search()
        return self._get_all_validation_raw_features(trace_id, self.search_task_id)

    @lru_cache()
    def _get_all_validation_raw_features(self, trace_id: str, search_task_id: str) -> Optional[pd.DataFrame]:
        time.sleep(1)  # this is neccesary to avoid requests rate limit restrictions
        features_response = get_rest_client(self.endpoint, self.api_key).get_search_features_v2(
            trace_id, search_task_id
        )
        result_df = None
        for feature_block in features_response["adsSearchTaskFeaturesDTO"]:
            if feature_block["searchType"] == "VALIDATION":
                features_id = feature_block["adsSearchTaskFeaturesId"]
                features_df = self._download_features_file(trace_id, features_id)
                if result_df is None:
                    result_df = features_df
                else:
                    result_df = pd.merge(result_df, features_df, how="outer", on=SYSTEM_RECORD_ID)

        return result_df

    def _get_features(self, trace_id: str, provider_summaries: List[ProviderTaskSummary]) -> List[Dict[str, Any]]:
        features = []
        file_metadata = self.get_file_metadata(trace_id)

        def get_original_column_name(name: str) -> str:
            for column_metadata in file_metadata.columns:
                if column_metadata.name == name:
                    return column_metadata.originalName or name
            return name

        for provider_summary in provider_summaries:
            if provider_summary.status == "COMPLETED":
                provider_features = get_rest_client(self.endpoint, self.api_key).get_search_features_meta_v2(
                    trace_id, provider_summary.ads_search_task_id
                )
                for feature in provider_features["providerFeatures"] + provider_features["etalonFeatures"]:
                    feature_meta = {
                        "feature_name": get_original_column_name(feature["name"]),
                        "shap_value": feature["importance"],
                        "coverage %": feature["matchedInPercent"],
                        "type": feature["valueType"] if "valueType" in feature else None,
                    }
                    features.append(feature_meta)
        return features

    def initial_features(self, trace_id: str) -> List[Dict[str, Any]]:
        provider_summaries = self._check_finished_initial_search()
        return self._get_features(trace_id, provider_summaries)

    def validation_features(self, trace_id: str) -> List[Dict[str, Any]]:
        provider_summaries = self._check_finished_validation_search()
        return self._get_features(trace_id, provider_summaries)

    def get_file_metadata(self, trace_id: str) -> FileMetadata:
        return get_rest_client(self.endpoint, self.api_key).get_search_file_metadata(self.search_task_id, trace_id)
