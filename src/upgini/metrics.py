from sklearn.model_selection import cross_val_predict
from sklearn.metrics import get_scorer
from sklearn.utils.multiclass import type_of_target
from catboost import CatBoostClassifier, CatBoostRegressor
from typing import List, Tuple, Union, Callable
import pandas as pd
from pandas.api.types import is_numeric_dtype


def calculate_cv_metric(X: pd.DataFrame, y, cv=5, scoring: Union[str, Callable, None] = None) -> float:
    cat_features_idx, cat_features = get_cat_features(X)
    X[cat_features].fillna("", inplace=True)

    target_type = type_of_target(y)
    if target_type in ["multiclass", "binary"]:
        estimator = CatBoostClassifier()
    elif target_type == "continuous":
        estimator = CatBoostRegressor()
    else:
        raise Exception(f"Unsupported type of target: {target_type}")

    scores = cross_val_predict(estimator, X, y, cv=cv, fit_params={"cat_features": cat_features_idx})
    metric_function = get_scorer(scoring)._score_func  # type: ignore
    return metric_function(y, scores)


def get_cat_features(X: pd.DataFrame) -> Tuple[List[int], List[str]]:
    zipped = [(i, c) for i, c in enumerate(X.columns) if not is_numeric_dtype(X[c])]
    unzipped = list(zip(*zipped))
    return unzipped[0], unzipped[1]  # type: ignore
