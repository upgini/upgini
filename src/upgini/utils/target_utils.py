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


def define_task(y: pd.Series, logger: Optional[logging.Logger] = None, silent: bool = False) -> ModelTaskType:
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
    target_ratio = target_items / len(target)
    if target_items == 2:
        task = ModelTaskType.BINARY
    elif (target.dtype.kind == "f" and np.any(target != target.astype(int))) or (
        is_numeric_dtype(target) and (target_items > 50 or target_ratio > 0.2)
    ):
        task = ModelTaskType.REGRESSION
    else:
        task = ModelTaskType.MULTICLASS
    logger.info(f"Detected task type: {task}")
    if not silent:
        print(bundle.get("target_type_detected").format(task))
    return task
