import numpy as np
import pandas as pd
import pytest

from upgini.errors import ValidationError
from upgini.metadata import ModelTaskType
from upgini.resource_bundle import bundle
from upgini.utils.target_utils import define_task


def test_invalid_target():
    y = pd.Series(["", "", ""])
    with pytest.raises(ValidationError, match=bundle.get("empty_target")):
        define_task(y)

    y = pd.Series([np.nan, np.inf, -np.inf])
    with pytest.raises(ValidationError, match=bundle.get("empty_target")):
        define_task(y)

    y = pd.Series([1, 1, 1, 1, 1])
    with pytest.raises(ValidationError, match=bundle.get("dataset_constant_target")):
        define_task(y)


def test_define_binary_task_type():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    assert define_task(y, False) == ModelTaskType.BINARY
    assert define_task(y, True) == ModelTaskType.BINARY

    y = pd.Series(["a", "b", "a", "b", "a"])
    assert define_task(y, False) == ModelTaskType.BINARY
    assert define_task(y, True) == ModelTaskType.BINARY


def test_define_multiclass_task_type():
    y = pd.Series(range(1, 51))
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series([float(x) for x in range(1, 51)])
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series(range(0, 50))
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series(["a", "b", "c", "b", "a"])
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series(["0", "1", "2", "3", "a"])
    assert define_task(y, False) == ModelTaskType.MULTICLASS
    assert define_task(y, True) == ModelTaskType.MULTICLASS

    y = pd.Series([0.0, 3.0, 5.0, 0.0, 5.0, 0.0, 3.0, 0.0, 5.0, 0.0, 5.0, 0.0, 3.0, 0.0, 3.0, 5.0, 3.0])
    assert define_task(y, False) == ModelTaskType.MULTICLASS


def test_define_regression_task_type():
    y = pd.Series([0.0, 3.0, 5.0, 0.0, 5.0, 0.0, 3.0, 0.0, 5.0, 0.0, 5.0, 0.0, 3.0, 0.0, 3.0, 5.0, 3.0])
    assert define_task(y, True) == ModelTaskType.REGRESSION

    y = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5])
    assert define_task(y, False) == ModelTaskType.REGRESSION
    assert define_task(y, True) == ModelTaskType.REGRESSION

    y = pd.Series([0, 1, 2, 3, 4, 5, 6, 8])
    assert define_task(y, False) == ModelTaskType.REGRESSION
    assert define_task(y, True) == ModelTaskType.REGRESSION

    y = pd.Series([0.0, 3.0, 5.0, 0.0, 5.0, 0.0, 3.0])
    assert define_task(y, False) == ModelTaskType.REGRESSION
    assert define_task(y, True) == ModelTaskType.REGRESSION
