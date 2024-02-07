from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold, TimeSeriesSplit, GroupKFold, GroupShuffleSplit

from upgini.metadata import CVType
from upgini.utils.blocked_time_series import BlockedTimeSeriesSplit


class CVConfig:
    def __init__(
        self,
        cv_type: Union[CVType, str, None],
        date_column: Optional[pd.Series],
        random_state=None,
        shuffle_kfold: Optional[bool] = None,
        test_size: Optional[float] = 0.2,
        n_folds: Optional[int] = 5,
        group_columns: Optional[List[str]] = None,
    ):
        if cv_type is None:
            self.cv_type = CVType.k_fold
        elif isinstance(cv_type, str):
            self.cv_type = CVType(cv_type)
        elif isinstance(cv_type, CVType):
            self.cv_type = cv_type
        else:
            raise Exception(f"Unexpected type of cv_type: {type(cv_type)}")

        self.group_columns = group_columns
        self.shuffle_kfold: Optional[bool] = shuffle_kfold
        self.test_size = test_size
        self.n_folds = n_folds
        if (self.cv_type == CVType.k_fold or self.cv_type == CVType.group_k_fold) and self.shuffle_kfold is None:
            self.shuffle_kfold = date_column is None or is_constant(date_column)
        if self.shuffle_kfold:
            self.random_state = random_state
        else:
            self.random_state = None

    def to_dict(self) -> Dict[str, Any]:
        config = {
            "cv_type": self.cv_type.value,
            "n_folds": self.n_folds,
        }
        if self.shuffle_kfold is not None:
            config["shuffle_kfold"] = self.shuffle_kfold
        if self.test_size is not None:
            config["test_size"] = self.test_size
        return config

    def get_cv_and_groups(self, X: pd.DataFrame) -> Tuple[BaseCrossValidator, Optional[np.ndarray]]:
        if self.cv_type == CVType.time_series:
            return TimeSeriesSplit(n_splits=self.n_folds), None
        elif self.cv_type == CVType.blocked_time_series:
            return BlockedTimeSeriesSplit(n_splits=self.n_folds, test_size=self.test_size), None
        elif self.cv_type == CVType.group_k_fold and self.group_columns:
            groups = get_groups(X, self.group_columns)

            if groups is None or np.unique(groups).size < self.n_folds:
                return KFold(n_splits=self.n_folds, shuffle=self.shuffle_kfold, random_state=self.random_state), None

            if self.shuffle_kfold:
                return (
                    GroupShuffleSplit(n_splits=self.n_folds, test_size=self.test_size, random_state=self.random_state),
                    groups,
                )
            else:
                return GroupKFold(n_splits=self.n_folds), groups
        else:
            return KFold(n_splits=self.n_folds, shuffle=self.shuffle_kfold, random_state=self.random_state), None


def get_groups(X: pd.DataFrame, group_columns: Optional[List[str]]) -> Optional[np.ndarray]:
    existing_group_columns = [c for c in group_columns if c in X.columns]
    return (
        None
        if not group_columns
        else reduce(
            lambda left, right: left + "_" + right, [X[c].astype(str) for c in existing_group_columns]
        ).factorize()[0]
    )


def is_constant(s, dropna=True) -> bool:
    if dropna:
        s = s[pd.notna(s)]
    a = s.to_numpy()
    return True if a.size == 0 else bool((a[0] == a).all())
