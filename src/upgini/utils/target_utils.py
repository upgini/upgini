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

    # Replace inf and -inf with NaN to handle extreme values correctly
    y = y.replace([np.inf, -np.inf], np.nan, inplace=False)

    # Drop NaN values from the target
    target = y.dropna()

    # Check if target is numeric and finite
    if is_numeric_dtype(target):
        target = target.loc[np.isfinite(target)]
    else:
        # If not numeric, drop empty strings as well
        target = target.loc[target != ""]

    # Raise error if there are no valid values left in the target
    if len(target) == 0:
        raise ValidationError(bundle.get("empty_target"))

    # Count unique values in the target
    target_items = target.nunique()

    # Raise error if all target values are the same
    if target_items == 1:
        raise ValidationError(bundle.get("dataset_constant_target"))

    reason = ""  # Will store the reason for selecting the task type

    # Binary classification case: exactly two unique values
    if target_items == 2:
        task = ModelTaskType.BINARY
        reason = bundle.get("binary_target_reason")
    else:
        # Attempt to convert target to numeric
        try:
            target = pd.to_numeric(target)
            is_numeric = True
        except Exception:
            is_numeric = False

        # If target cannot be converted to numeric, assume multiclass classification
        if not is_numeric:
            task = ModelTaskType.MULTICLASS
            reason = bundle.get("non_numeric_multiclass_reason")
        else:
            # Multiclass classification: few unique values and integer encoding
            if target.nunique() <= 50 and is_int_encoding(target.unique()):
                task = ModelTaskType.MULTICLASS
                reason = bundle.get("few_unique_label_multiclass_reason")
            # Regression case: if there is date, assume regression
            elif has_date:
                task = ModelTaskType.REGRESSION
                reason = bundle.get("date_search_key_regression_reason")
            else:
                # Remove zero values and recalculate unique ratio
                non_zero_target = target[target != 0]
                target_items = non_zero_target.nunique()
                target_ratio = target_items / len(non_zero_target)

                # Use unique_ratio to determine whether to classify as regression or multiclass
                if (
                    (target.dtype.kind == "f" and np.any(target != target.astype(int)))  # Non-integer float values
                    or target_items > 50
                    or target_ratio > 0.2  # If non-zero values have high ratio of uniqueness
                ):
                    task = ModelTaskType.REGRESSION
                    reason = bundle.get("many_unique_label_regression_reason")
                else:
                    task = ModelTaskType.MULTICLASS
                    reason = bundle.get("limited_int_multiclass_reason")

    # Log or print the reason for the selected task type
    logger.info(f"Detected task type: {task} (Reason: {reason})")

    # Print task type and reason if silent mode is off
    if not silent:
        print(bundle.get("target_type_detected").format(task, reason))

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
    binary_min_sample_threshold: int = 5000,
    multiclass_min_sample_threshold: int = 25000,
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

    # count = len(df)
    target = df[target_column].copy()
    # target_classes_count = target.nunique()

    vc = target.value_counts()
    max_class_value = vc.index[0]
    min_class_value = vc.index[len(vc) - 1]
    max_class_count = vc[max_class_value]
    min_class_count = vc[min_class_value]
    num_classes = len(vc)

    # min_class_percent = imbalance_threshold / target_classes_count
    # min_class_threshold = int(min_class_percent * count)

    resampled_data = df
    df = df.copy().sort_values(by=SYSTEM_RECORD_ID)
    if task_type == ModelTaskType.MULTICLASS:
        if len(df) > multiclass_min_sample_threshold and max_class_count > (
            min_class_count * multiclass_bootstrap_loops
        ):

            # msg = bundle.get("imbalance_multiclass").format(min_class_value, min_class_count)
            msg = bundle.get("imbalanced_target").format(min_class_value, min_class_count)
            logger.warning(msg)
            print(msg)
            if warning_counter:
                warning_counter.increment()

            sample_strategy = dict()
            for class_value in vc.index:
                if class_value == min_class_value:
                    continue
                class_count = vc[class_value]
                sample_size = min(
                    class_count,
                    multiclass_bootstrap_loops
                    * (
                        min_class_count
                        + max((multiclass_min_sample_threshold - num_classes * min_class_count) / (num_classes - 1), 0)
                    ),
                )
                sample_strategy[class_value] = int(sample_size)
            logger.info(f"Rebalance sample strategy: {sample_strategy}. Min class count: {min_class_count}")
            sampler = RandomUnderSampler(sampling_strategy=sample_strategy, random_state=random_state)
            X = df[SYSTEM_RECORD_ID]
            X = X.to_frame(SYSTEM_RECORD_ID)
            new_x, _ = sampler.fit_resample(X, target)  # type: ignore

            resampled_data = df[df[SYSTEM_RECORD_ID].isin(new_x[SYSTEM_RECORD_ID])]
    elif len(df) > binary_min_sample_threshold:
        # msg = bundle.get("dataset_rarest_class_less_threshold").format(
        #     min_class_value, min_class_count, min_class_threshold, min_class_percent * 100
        # )
        msg = bundle.get("imbalanced_target").format(min_class_value, min_class_count)
        logger.warning(msg)
        print(msg)
        if warning_counter:
            warning_counter.increment()

        # fill up to min_sample_threshold by majority class
        minority_class = df[df[target_column] == min_class_value]
        majority_class = df[df[target_column] != min_class_value]
        # sample_size = min(len(majority_class), min_sample_threshold - min_class_count)
        sample_size = min(
            max_class_count,
            binary_bootstrap_loops * (min_class_count + max(binary_min_sample_threshold - 2 * min_class_count, 0)),
        )
        logger.info(
            f"Min class count: {min_class_count}. Max class count: {max_class_count}."
            f" Rebalance sample size: {sample_size}"
        )
        sampled_majority_class = majority_class.sample(n=sample_size, random_state=random_state)
        resampled_data = df[
            (df[SYSTEM_RECORD_ID].isin(minority_class[SYSTEM_RECORD_ID]))
            | (df[SYSTEM_RECORD_ID].isin(sampled_majority_class[SYSTEM_RECORD_ID]))
        ]

    # elif max_class_count > min_class_count * binary_bootstrap_loops:
    #     msg = bundle.get("dataset_rarest_class_less_threshold").format(
    #         min_class_value, min_class_count, min_class_threshold, min_class_percent * 100
    #     )
    #     logger.warning(msg)
    #     print(msg)
    #     if warning_counter:
    #         warning_counter.increment()

    #     sampler = RandomUnderSampler(
    #         sampling_strategy={max_class_value: binary_bootstrap_loops * min_class_count}, random_state=random_state
    #     )
    #     X = df[SYSTEM_RECORD_ID]
    #     X = X.to_frame(SYSTEM_RECORD_ID)
    #     new_x, _ = sampler.fit_resample(X, target)  # type: ignore

    #     resampled_data = df[df[SYSTEM_RECORD_ID].isin(new_x[SYSTEM_RECORD_ID])]

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
    try:
        return np.sum((train_distribution - test_distribution) * np.log(train_distribution / test_distribution))
    except Exception:
        return np.nan
