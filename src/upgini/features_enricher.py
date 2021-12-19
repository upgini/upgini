from datetime import date
from typing import Dict, List, Optional, Union

try:
    from sklearn.base import TransformerMixin  # type: ignore
    from sklearn.exceptions import NotFittedError  # type: ignore
except ImportError:
    TransformerMixin = object
    NotFittedError = Exception


import itertools
import uuid

import numpy as np
import pandas as pd

from upgini.dataset import Dataset
from upgini.metadata import (
    SYSTEM_FAKE_DATE,
    SYSTEM_RECORD_ID,
    FileColumnMeaningType,
    ModelTaskType,
    SearchKey,
)
from upgini.search_task import SearchTask
from upgini.utils.format import Format


class FeaturesEnricher(TransformerMixin):  # type: ignore
    """Retrieve external features via Upgini that are most relevant to predict your target.

    Parameters
    ----------
    search_keys: dict of str->SearchKey or int->SearchKey
        Dictionary with column names or indices mapping to key types.
        Each of this columns will be used as a search key to find features.

    keep_input: bool, optional (default=False)
        If True, copy original input columns to the output dataframe.

    accurate_model: bool, optional (default=False)
        If True, search takes longer but returned metrics may be more accurate.

    api_key: str, optional (default=None)
        Token to authorize search requests. You can get it on https://profile.upgini.com/.
        If not specified then read the value from the environment variable UPGINI_API_KEY.

    endpoint: str, optional (default=None)
        URL of Upgini API where search requests are submitted.
        If not specified then used the default value.
        Please don't overwrite it if you are unsure.
    """

    TARGET_NAME = "target"
    EVAL_SET_INDEX = "eval_set_index"

    _search_task: Optional[SearchTask] = None
    passed_features: List[str] = []

    def __init__(
        self,
        search_keys: Union[Dict[str, SearchKey], Dict[int, SearchKey]],
        keep_input: bool = False,
        accurate_model: bool = False,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        if len(search_keys) == 0:
            raise ValueError("Key columns should be marked up by search_keys.")
        self.search_keys = search_keys
        self.keep_input = keep_input
        self.accurate_model = accurate_model
        self.endpoint = endpoint
        self.api_key = api_key

    def _inner_fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list] = None,
        eval_set: Optional[List[tuple]] = None,
        extract_features: bool = False,
        **fit_params,
    ) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Only pandas.DataFrame supported for X, but {type(X)} was passed.")
        if not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not isinstance(y, list):
            raise TypeError(f"Only pandas.Series or numpy.ndarray or list supported for y, but {type(y)} was passed.")

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        if X.shape[0] != len(y_array):
            raise ValueError("X and y should be the same size")

        validated_search_keys = self._prepare_search_keys(X)

        search_keys = []
        for L in range(1, len(validated_search_keys.keys()) + 1):
            for subset in itertools.combinations(validated_search_keys.keys(), L):
                search_keys.append(subset)
        meaning_types = {
            **validated_search_keys.copy(),
            **{str(c): FileColumnMeaningType.FEATURE for c in X.columns if c not in validated_search_keys.keys()},
        }

        df = X.copy()
        df[self.TARGET_NAME] = y_array

        df.reset_index(drop=True, inplace=True)

        meaning_types[self.TARGET_NAME] = FileColumnMeaningType.TARGET

        df[SYSTEM_RECORD_ID] = df.apply(lambda row: hash(tuple(row)), axis=1)
        meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID

        df_without_eval_set = df.copy()

        if eval_set is not None and len(eval_set) > 0:
            df[self.EVAL_SET_INDEX] = 0
            meaning_types[self.EVAL_SET_INDEX] = FileColumnMeaningType.EVAL_SET_INDEX
            for idx, eval_pair in enumerate(eval_set):
                if len(eval_pair) != 2:
                    raise TypeError(
                        f"Invalid size of eval_set pair: {len(eval_pair)}. "
                        "It should contain tuples of 2 elements: X and y."
                    )
                eval_X = eval_pair[0]
                eval_y = eval_pair[1]
                if not isinstance(eval_X, pd.DataFrame):
                    raise TypeError(
                        f"Only pandas.DataFrame supported for X in eval_set, but {type(eval_X)} was passed."
                    )
                if (
                    not isinstance(eval_y, pd.Series)
                    and not isinstance(eval_y, np.ndarray)
                    and not isinstance(eval_y, list)
                ):
                    raise TypeError(
                        "pandas.Series or numpy.ndarray or list supported for y in eval_set, "
                        f"but {type(eval_y)} was passed."
                    )
                eval_df = eval_X.copy()
                eval_df[self.TARGET_NAME] = pd.Series(eval_y)
                eval_df[SYSTEM_RECORD_ID] = eval_df.apply(lambda row: hash(tuple(row)), axis=1)
                eval_df[self.EVAL_SET_INDEX] = idx + 1
                df = pd.concat([df, eval_df], ignore_index=True)

        if FileColumnMeaningType.DATE not in meaning_types.values():
            df[SYSTEM_FAKE_DATE] = date.today()
            search_keys.append((SYSTEM_FAKE_DATE,))
            meaning_types[SYSTEM_FAKE_DATE] = FileColumnMeaningType.DATE

        dataset = Dataset("tds_" + str(uuid.uuid4()), df=df, endpoint=self.endpoint, api_key=self.api_key)
        dataset.meaning_types = meaning_types
        dataset.search_keys = search_keys

        self.passed_features = [
            column for column, meaning_type in meaning_types.items() if meaning_type == FileColumnMeaningType.FEATURE
        ]

        self._search_task = dataset.search(extract_features=extract_features, accurate_model=self.accurate_model)
        metrics = self.get_metrics()
        if metrics is not None:
            print(Format.GREEN + Format.BOLD + "\nQuality metrics" + Format.END)
            try:
                from IPython.display import display  # type: ignore

                display(metrics)
            except ImportError:
                print(metrics)

            if self.__is_uplift_present_in_metrics():
                print(
                    "\nFollowing features was used for accuracy uplift estimation:",
                    ", ".join(self.passed_features),
                )

        return df_without_eval_set

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, List],
        eval_set: Optional[List[tuple]] = None,
        **fit_params,
    ):
        self._inner_fit(X, y, eval_set, False, **fit_params)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, List],
        eval_set: Optional[List[tuple]] = None,
        **fit_params,
    ) -> pd.DataFrame:
        df = self._inner_fit(X, y, eval_set, extract_features=True, **fit_params)

        etalon_columns = X.columns + self.TARGET_NAME

        if self._search_task is None:
            raise RuntimeError("Fit wasn't completed successfully.")

        print("Executing transform step...")
        result_features = self._search_task.get_all_initial_raw_features()

        if result_features is None:
            raise RuntimeError("No provider supports feature enrichment.")

        if self.keep_input:
            result = pd.merge(
                df.drop(columns=self.TARGET_NAME),
                result_features,
                left_on=SYSTEM_RECORD_ID,
                right_on=SYSTEM_RECORD_ID,
                how="left",
            )
        else:
            result = pd.merge(df, result_features, left_on=SYSTEM_RECORD_ID, right_on=SYSTEM_RECORD_ID, how="left")
            result.drop(columns=etalon_columns, inplace=True)

        result.index = X.index
        if SYSTEM_RECORD_ID in result.columns:
            result.drop(columns=SYSTEM_RECORD_ID, inplace=True)
        if SYSTEM_FAKE_DATE in result.columns:
            result.drop(columns=SYSTEM_FAKE_DATE, inplace=True)
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._search_task is None:
            raise NotFittedError("`fit` or `fit_transform` should be called before `transform`.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Only pandas.DataFrame supported for X, but {type(X)} was passed.")

        validated_search_keys = self._prepare_search_keys(X)
        search_keys = []
        for L in range(1, len(validated_search_keys.keys()) + 1):
            for subset in itertools.combinations(validated_search_keys.keys(), L):
                search_keys.append(subset)
        meaning_types = validated_search_keys.copy()
        feature_columns = [column for column in X.columns if column not in meaning_types.keys()]

        df = X.copy()

        df = df.reset_index(drop=True)

        if FileColumnMeaningType.DATE not in meaning_types.values():
            df[SYSTEM_FAKE_DATE] = date.today()
            search_keys.append((SYSTEM_FAKE_DATE,))
            meaning_types[SYSTEM_FAKE_DATE] = FileColumnMeaningType.DATE

        df[SYSTEM_RECORD_ID] = df.apply(lambda row: hash(tuple(row[meaning_types.keys()])), axis=1)
        meaning_types[SYSTEM_RECORD_ID] = FileColumnMeaningType.SYSTEM_RECORD_ID

        # Don't pass features in backend on transform
        if feature_columns:
            df_without_features = df.drop(columns=feature_columns)
        else:
            df_without_features = df

        dataset = Dataset(
            "sample_" + str(uuid.uuid4()), df=df_without_features, endpoint=self.endpoint, api_key=self.api_key
        )
        dataset.meaning_types = meaning_types
        dataset.search_keys = search_keys
        validation_task = self._search_task.validation(dataset, extract_features=True)

        etalon_columns = list(self.search_keys.keys())

        print("Executing transform step...")
        result_features = validation_task.get_all_validation_raw_features()

        if result_features is None:
            raise RuntimeError("No provider supports feature enrichment.")

        if not self.keep_input:
            result = pd.merge(
                df_without_features, result_features, left_on=SYSTEM_RECORD_ID, right_on=SYSTEM_RECORD_ID, how="left"
            )
            result.drop(columns=etalon_columns, inplace=True)
        else:
            result = pd.merge(df, result_features, left_on=SYSTEM_RECORD_ID, right_on=SYSTEM_RECORD_ID, how="left")

        result.index = X.index
        if SYSTEM_RECORD_ID in result.columns:
            result.drop(columns=SYSTEM_RECORD_ID, inplace=True)
        if SYSTEM_FAKE_DATE in result.columns:
            result.drop(columns=SYSTEM_FAKE_DATE, inplace=True)
        return result

    def _prepare_search_keys(self, x: pd.DataFrame) -> Dict[str, FileColumnMeaningType]:
        valid_search_keys = {}
        for column_id, meaning_type in self.search_keys.items():
            if isinstance(column_id, str):
                valid_search_keys[column_id] = meaning_type.value
            elif isinstance(column_id, int):
                valid_search_keys[x.columns[column_id]] = meaning_type.value
            else:
                raise ValueError(f"Unsupported type of key in search_keys: {type(column_id)}.")
            if meaning_type == SearchKey.CUSTOM_KEY:
                raise ValueError("SearchKey.CUSTOM_KEY is not supported for FeaturesEnricher.")

        return valid_search_keys

    def get_metrics(self) -> Optional[pd.DataFrame]:
        if self._search_task is None or self._search_task.summary is None:
            raise NotFittedError("Run fit before get metrics.")

        metrics = []
        initial_metrics = {}

        initial_hit_rate = self._search_task.initial_max_hit_rate()
        if initial_hit_rate is None:
            return None

        initial_metrics["segment"] = "train"
        initial_metrics["match rate"] = initial_hit_rate["value"]
        metrics.append(initial_metrics)
        initial_auc = self._search_task.initial_max_auc()
        if initial_auc is not None:
            initial_metrics["auc"] = initial_auc["value"]
        initial_accuracy = self._search_task.initial_max_accuracy()
        if initial_accuracy is not None:
            initial_metrics["accuracy"] = initial_accuracy["value"]
        initial_rmse = self._search_task.initial_max_rmse()
        if initial_rmse is not None:
            initial_metrics["rmse"] = initial_rmse["value"]
        initial_uplift = self._search_task.initial_max_uplift()
        if len(self.passed_features) > 0 and initial_uplift is not None:
            initial_metrics["uplift"] = initial_uplift["value"]
        max_initial_eval_set_metrics = self._search_task.get_max_initial_eval_set_metrics()

        if max_initial_eval_set_metrics is not None:
            for eval_set_metrics in max_initial_eval_set_metrics:
                if "gini" in eval_set_metrics:
                    del eval_set_metrics["gini"]
                eval_set_index = eval_set_metrics["eval_set_index"]
                eval_set_metrics["match rate"] = eval_set_metrics["hit_rate"]
                eval_set_metrics["segment"] = f"eval {eval_set_index}"
                del eval_set_metrics["hit_rate"]
                del eval_set_metrics["eval_set_index"]
                metrics.append(eval_set_metrics)

        metrics_df = pd.DataFrame(metrics)
        metrics_df.set_index("segment", inplace=True)
        metrics_df.rename_axis(None, inplace=True)
        return metrics_df

    def __is_uplift_present_in_metrics(self):
        uplift_presented = False

        if self._search_task is not None and self._search_task.summary is not None:
            if len(self.passed_features) > 0 and self._search_task.initial_max_uplift() is not None:
                uplift_presented = True

            max_initial_eval_set_metrics = self._search_task.get_max_initial_eval_set_metrics()

            if max_initial_eval_set_metrics is not None:
                for eval_set_metrics in max_initial_eval_set_metrics:
                    if "uplift" in eval_set_metrics:
                        uplift_presented = True

            if self._search_task.summary.status == "VALIDATION_COMPLETED":
                if len(self.passed_features) > 0 and self._search_task.validation_max_uplift() is not None:
                    uplift_presented = True

        return uplift_presented

    def __is_quality_by_metrics_low(self) -> bool:
        if self._search_task is None:
            return False
        if len(self.passed_features) > 0 and self._search_task.task_type is not None:
            max_uplift = self._search_task.initial_max_uplift()
            if self._search_task.task_type == ModelTaskType.BINARY:
                threshold = 0.002
            elif self._search_task.task_type == ModelTaskType.MULTICLASS:
                threshold = 3.0
            elif self._search_task.task_type == ModelTaskType.REGRESSION:
                threshold = 0.0
            else:
                return False
            if max_uplift is not None and max_uplift["value"] < threshold:
                return True
        elif self._search_task.task_type is not None:
            max_auc = self._search_task.initial_max_auc()
            if self._search_task.task_type == ModelTaskType.BINARY and max_auc is not None:
                if max_auc["value"] < 0.55:
                    return True
        return False

    def __check_quality(self, no_data_found: bool):
        if no_data_found or self.__is_quality_by_metrics_low():
            try:
                from IPython.display import HTML, display  # type: ignore

                display(
                    HTML(
                        "<h9>Oops, looks like we're not able to find data which gives a strong uplift for your ML "
                        "algorithm.<br>If you have ANY data which you might consider as royalty and "
                        "license-free and potentially valuable for supervised ML applications,<br> we shall be "
                        "happy to give you free individual access in exchange for sharing this data with "
                        "community.<br>Just upload your data sample right from Jupyter. We will check your data "
                        "sharing proposal and get back to you ASAP."
                    )
                )
                display(
                    HTML(
                        "<a href='https://github.com/upgini/upgini/blob/main/README.md"
                        "#share-license-free-data-with-community' "
                        "target='_blank'>How to upload your data sample from Jupyter</a>"
                    )
                )
            except ImportError:
                print("Oops, looks like we're not able to find data which gives a strong uplift for your ML algorithm.")
                print(
                    "If you have ANY data which you might consider as royalty and license-free and potentially "
                    "valuable for supervised ML applications,"
                )
                print(
                    "we shall be happy to give you free individual access in exchange for sharing this data with "
                    "community."
                )
                print(
                    "Just upload your data sample right from Jupyter. We will check your data sharing proposal and "
                    "get back to you ASAP."
                )
                print("https://github.com/upgini/upgini/blob/main/README.md#share-license-free-data-with-community")
