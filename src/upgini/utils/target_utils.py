import logging
from typing import Union

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


def define_task(y: pd.Series, logger: logging.Logger, silent: bool = False) -> ModelTaskType:
    target = y.dropna()
    if is_numeric_dtype(target):
        target = target.loc[np.isfinite(target)]
    else:
        target = target.loc[target != ""]
    if len(target) == 0:
        raise ValidationError(bundle.get("empty_target"))
    target_items = target.nunique()
    target_ratio = target_items / len(target)
    if (target_items > 50 or (target_items > 2 and target_ratio > 0.2)) and is_numeric_dtype(target):
        task = ModelTaskType.REGRESSION
    elif target_items <= 2:
        task = ModelTaskType.BINARY
    else:
        task = ModelTaskType.MULTICLASS
    logger.info(f"Detected task type: {task}")
    if not silent:
        print(bundle.get("target_type_detected").format(task))
    return task
