"""
Utility functions for autofe module.
"""

import functools
from typing import Callable, Union

import numpy as np
from pydantic import BaseModel


def get_pydantic_version():
    """
    Get the major version of pydantic.

    Returns:
        int: Major version number (1 or 2)
    """
    try:
        from pydantic import __version__ as pydantic_version

        major_version = int(pydantic_version.split(".")[0])
        return major_version
    except (ImportError, ValueError):
        # Default to version 1 if unable to determine
        return 1


def pydantic_validator(field_name: str, *fields, mode: str = "before", **kwargs):
    """
    A decorator that applies the appropriate Pydantic validator based on the installed version.

    This decorator handles the differences between Pydantic v1 and v2 validator syntax,
    making it easier to write code that works with both versions.

    Args:
        field_name (str): The name of the field to validate
        mode (str): The validation mode, either "before" or "after" (for Pydantic v2)
        **kwargs: Additional arguments to pass to the validator

    Returns:
        Callable: A decorator that can be applied to validator methods

    Example:
        ```python
        class MyModel(BaseModel):
            items: List[int]

            @pydantic_validator("items")
            def parse_items(cls, value):
                if isinstance(value, str):
                    return [int(x) for x in value.split(",")]
                return value
        ```
    """
    pydantic_version = get_pydantic_version()

    if pydantic_version >= 2:
        # Use field_validator for Pydantic 2.x
        from pydantic import field_validator

        def decorator(func: Callable) -> Callable:
            @field_validator(field_name, *fields, mode=mode, **kwargs)
            @functools.wraps(func)
            def wrapper(cls, value, **kw):
                return func(cls, value)

            return wrapper

        return decorator
    else:
        # Use validator for Pydantic 1.x
        from pydantic import validator

        # Map mode to Pydantic v1 parameters
        pre = True if mode == "before" else False

        def decorator(func: Callable) -> Callable:
            @validator(field_name, *fields, pre=pre, **kwargs)
            @functools.wraps(func)
            def wrapper(cls, value, **kw):
                return func(cls, value)

            return wrapper

        return decorator


def pydantic_json_method(obj: BaseModel) -> str:
    if get_pydantic_version() >= 2:
        return obj.model_dump_json
    else:
        return obj.json


def pydantic_parse_method(cls):
    if get_pydantic_version() >= 2:
        return cls.model_validate
    else:
        return cls.parse_obj


def pydantic_dump_method(obj: BaseModel) -> dict:
    if get_pydantic_version() >= 2:
        return obj.model_dump
    else:
        return obj.dict


def pydantic_copy_method(obj):
    if get_pydantic_version() >= 2:
        return obj.model_copy
    else:
        return obj.copy


def bin_index(value: Union[float, int, None], bounds) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    bounds_arr = np.asarray(bounds, dtype=np.float64)
    if bounds_arr.size == 0 or value < bounds_arr[0]:
        return np.nan
    return np.searchsorted(bounds_arr, value, side="right")


def bin_index_vectorized(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    n = len(values)
    result = np.full(n, np.nan)
    bounds_arr = np.asarray(bounds, dtype=np.float64)
    if bounds_arr.size == 0:
        return result
    valid = ~np.isnan(values)
    if not valid.any():
        return result
    valid_values = values[valid]
    idx = np.searchsorted(bounds_arr, valid_values, side="right").astype(np.float64)
    below = valid_values < bounds_arr[0]
    if below.any():
        idx[below] = np.nan
    result[valid] = idx
    return result


def bin_index_many(values: np.ndarray, bounds_2d: np.ndarray) -> np.ndarray:
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(n):
        v = values[i]
        if np.isnan(v):
            continue
        bounds_row = bounds_2d[i]
        if bounds_row.size == 0 or v < bounds_row[0]:
            continue
        result[i] = np.searchsorted(bounds_row, v, side="right")
    return result
