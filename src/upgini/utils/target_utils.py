import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import numpy as np
import logging

from upgini.metadata import ModelTaskType
from upgini.errors import ValidationError
from upgini.resource_bundle import bundle


def correct_target(y: pd.Series) -> pd.Series:
    if is_string_dtype(y):
        unique_target = y.unique()
        target_replacement = {v: i for i, v in enumerate(unique_target)}
        return y.replace(target_replacement)
    else:
        return y


def define_task(y: pd.Series, logger: logging.Logger, silent: bool = False) -> ModelTaskType:
    target = y.dropna()
    if is_numeric_dtype(target):
        target = target.loc[np.isfinite(target)]  # type: ignore
    else:
        target = target.loc[target != ""]
    if len(target) == 0:
        raise ValidationError(bundle.get("empty_target"))
    target_items = target.nunique()
    target_ratio = target_items / len(target)
    if (target_items > 50 or (target_items > 2 and target_ratio > 0.2)) and is_numeric_dtype(target):
        task = ModelTaskType.REGRESSION
    elif target_items <= 2:
        if is_numeric_dtype(target):
            task = ModelTaskType.BINARY
        else:
            raise ValidationError(bundle.get("non_numeric_target"))
    else:
        task = ModelTaskType.MULTICLASS
    logger.info(f"Detected task type: {task}")
    if not silent:
        print(bundle.get("target_type_detected").format(task))
    return task
