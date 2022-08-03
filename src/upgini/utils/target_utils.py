import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import logging

from upgini.metadata import ModelTaskType


def define_task(y: pd.Series, logger: logging.Logger, silent: bool = False) -> ModelTaskType:
    target = y.dropna()
    if is_numeric_dtype(target):
        target = target.loc[np.isfinite(target)]  # type: ignore
    else:
        target = target.loc[target != ""]
    if len(target) == 0:
        raise ValueError("Target is empty in all rows")
    target_items = target.nunique()
    target_ratio = target_items / len(target)
    if (target_items > 50 or (target_items > 2 and target_ratio > 0.2)) and is_numeric_dtype(target):
        task = ModelTaskType.REGRESSION
    elif target_items <= 2:
        if is_numeric_dtype(target):
            task = ModelTaskType.BINARY
        else:
            raise ValueError("Binary target should be numerical")
    else:
        task = ModelTaskType.MULTICLASS
    logger.info(f"Detected task type: {task}")
    if not silent:
        print(f"Detected task type: {task}")
    return task
