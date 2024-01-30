import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from upgini.errors import ValidationError
from upgini.metadata import ModelTaskType
from upgini.resource_bundle import bundle


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
