import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from upgini.errors import ValidationError
from upgini.metadata import SYSTEM_RECORD_ID, ModelTaskType
from upgini.resource_bundle import ResourceBundle, bundle, get_custom_bundle
from upgini.sampler.random_under_sampler import RandomUnderSampler
from upgini.utils.warning_counter import WarningCounter


def correct_string_target(y: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    if isinstance(y, pd.Series):
        return y.astype(str).astype("category").cat.codes
    elif isinstance(y, np.ndarray):
        return pd.Series(y).astype(str).astype("category").cat.codes.values


def define_task(
    y: pd.Series, has_date: bool = False, logger: Optional[logging.Logger] = None, silent: bool = False
) -> ModelTaskType:
    if logger is None:
        logger = logging.getLogger()
    target = y.dropna()
    if is_numeric_dtype(target):
        target = target.loc[np.isfinite(target)]
    else:
        target = target.loc[target != ""]
    if len(target) == 0:
        raise ValidationError(bundle.get("empty_target"))
    target_items = target.nunique()
    if target_items == 1:
        raise ValidationError(bundle.get("dataset_constant_target"))
    if target_items == 2:
        task = ModelTaskType.BINARY
    else:
        try:
            target = pd.to_numeric(target)
            is_numeric = True
        except Exception:
            is_numeric = False

        # If any value is non numeric - multiclass
        if not is_numeric:
            task = ModelTaskType.MULTICLASS
        else:
            if target.nunique() <= 50 and is_int_encoding(target.unique()):
                task = ModelTaskType.MULTICLASS
            elif has_date:
                task = ModelTaskType.REGRESSION
            else:
                non_zero_target = target[target != 0]
                target_items = non_zero_target.nunique()
                target_ratio = target_items / len(non_zero_target)
                if (
                    (target.dtype.kind == "f" and np.any(target != target.astype(int)))  # any non integer
                    or target_items > 50
                    or target_ratio > 0.2
                ):
                    task = ModelTaskType.REGRESSION
                else:
                    task = ModelTaskType.MULTICLASS

    logger.info(f"Detected task type: {task}")
    if not silent:
        print(bundle.get("target_type_detected").format(task))
    return task


def is_int_encoding(unique_values):
    return set(unique_values) == set(range(len(unique_values))) or set(unique_values) == set(
        range(1, len(unique_values) + 1)
    )


def balance_undersample(
    df: pd.DataFrame,
    target_column: str,
    task_type: ModelTaskType,
    random_state: int,
    imbalance_threshold: int = 0.2,
    min_sample_threshold: int = 5000,
    binary_bootstrap_loops: int = 5,
    multiclass_bootstrap_loops: int = 2,
    logger: Optional[logging.Logger] = None,
    bundle: Optional[ResourceBundle] = None,
    warning_counter: Optional[WarningCounter] = None,
) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger("muted_logger")
        logger.setLevel("FATAL")
    bundle = bundle or get_custom_bundle()
    if SYSTEM_RECORD_ID not in df.columns:
        raise Exception("System record id must be presented for undersampling")

    count = len(df)
    target = df[target_column].copy()
    target_classes_count = target.nunique()

    vc = target.value_counts()
    max_class_value = vc.index[0]
    min_class_value = vc.index[len(vc) - 1]
    max_class_count = vc[max_class_value]
    min_class_count = vc[min_class_value]

    min_class_percent = imbalance_threshold / target_classes_count
    min_class_threshold = int(min_class_percent * count)

    resampled_data = df
    df = df.copy().sort_values(by=SYSTEM_RECORD_ID)
    if task_type == ModelTaskType.MULTICLASS:
        # Sort classes by rows count and find 25% quantile class
        classes = vc.index
        quantile25_idx = int(0.75 * len(classes)) - 1
        quantile25_class = classes[quantile25_idx]
        quantile25_class_cnt = vc[quantile25_class]

        if max_class_count > (quantile25_class_cnt * multiclass_bootstrap_loops):
            msg = bundle.get("imbalance_multiclass").format(quantile25_class, quantile25_class_cnt)
            logger.warning(msg)
            print(msg)
            if warning_counter:
                warning_counter.increment()

            # 25% and lower classes will stay as is. Higher classes will be downsampled
            sample_strategy = dict()
            for class_idx in range(quantile25_idx):
                # compare class count with count_of_quantile25_class * 2
                class_value = classes[class_idx]
                class_count = vc[class_value]
                sample_strategy[class_value] = min(class_count, quantile25_class_cnt * multiclass_bootstrap_loops)
            sampler = RandomUnderSampler(sampling_strategy=sample_strategy, random_state=random_state)
            X = df[SYSTEM_RECORD_ID]
            X = X.to_frame(SYSTEM_RECORD_ID)
            new_x, _ = sampler.fit_resample(X, target)  # type: ignore

            resampled_data = df[df[SYSTEM_RECORD_ID].isin(new_x[SYSTEM_RECORD_ID])]
    elif len(df) > min_sample_threshold and min_class_count < min_sample_threshold / 2:
        msg = bundle.get("dataset_rarest_class_less_threshold").format(
            min_class_value, min_class_count, min_class_threshold, min_class_percent * 100
        )
        logger.warning(msg)
        print(msg)
        if warning_counter:
            warning_counter.increment()

        # fill up to min_sample_threshold by majority class
        minority_class = df[df[target_column] == min_class_value]
        majority_class = df[df[target_column] != min_class_value]
        sample_size = min(len(majority_class), min_sample_threshold - min_class_count)
        sampled_majority_class = majority_class.sample(n=sample_size, random_state=random_state)
        resampled_data = df[
            (df[SYSTEM_RECORD_ID].isin(minority_class[SYSTEM_RECORD_ID]))
            | (df[SYSTEM_RECORD_ID].isin(sampled_majority_class[SYSTEM_RECORD_ID]))
        ]

    elif max_class_count > min_class_count * binary_bootstrap_loops:
        msg = bundle.get("dataset_rarest_class_less_threshold").format(
            min_class_value, min_class_count, min_class_threshold, min_class_percent * 100
        )
        logger.warning(msg)
        print(msg)
        if warning_counter:
            warning_counter.increment()

        sampler = RandomUnderSampler(
            sampling_strategy={max_class_value: binary_bootstrap_loops * min_class_count}, random_state=random_state
        )
        X = df[SYSTEM_RECORD_ID]
        X = X.to_frame(SYSTEM_RECORD_ID)
        new_x, _ = sampler.fit_resample(X, target)  # type: ignore

        resampled_data = df[df[SYSTEM_RECORD_ID].isin(new_x[SYSTEM_RECORD_ID])]

    logger.info(f"Shape after rebalance resampling: {resampled_data}")
    return resampled_data


def calculate_psi(expected: pd.Series, actual: pd.Series) -> float:
    df = pd.concat([expected, actual])

    # Define the bins for the target variable
    df_min = df.min()
    df_max = df.max()
    bins = [df_min, (df_min + df_max) / 2, df_max]

    # Calculate the base distribution
    train_distribution = expected.value_counts(bins=bins, normalize=True).sort_index().values

    # Calculate the target distribution
    test_distribution = actual.value_counts(bins=bins, normalize=True).sort_index().values

    # Calculate the PSI
    return np.sum((train_distribution - test_distribution) * np.log(train_distribution / test_distribution))
