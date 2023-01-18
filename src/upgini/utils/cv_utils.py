from typing import Any, Dict, Optional, Union

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold, TimeSeriesSplit

from upgini.metadata import CVType
from upgini.utils.blocked_time_series import BlockedTimeSeriesSplit


class CVConfig:
    def __init__(self, cv_type: Union[CVType, str, None], date_column: Optional[pd.Series], random_state=None):
        if cv_type is None:
            self.cv_type = CVType.k_fold
        elif isinstance(cv_type, str):
            self.cv_type = CVType(cv_type)
        elif isinstance(cv_type, CVType):
            self.cv_type = cv_type
        else:
            raise Exception(f"Unexcpected type of cv_type: {type(cv_type)}")

        self.shuffle_kfold: Optional[bool] = None
        self.test_size: Optional[float] = None
        if self.cv_type == CVType.k_fold:
            self.n_folds = 5
            self.shuffle_kfold = date_column is None or is_constant(date_column)
        elif self.cv_type == CVType.time_series or cv_type == CVType.blocked_time_series:
            self.n_folds = 10
            self.test_size = 0.2
        else:
            raise Exception(f"Unexpected cv_type: {cv_type}")
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

    def get_cv(self) -> BaseCrossValidator:
        if self.cv_type == CVType.time_series:
            return TimeSeriesSplit(n_splits=self.n_folds)
        elif self.cv_type == CVType.blocked_time_series:
            return BlockedTimeSeriesSplit(n_splits=self.n_folds, test_size=self.test_size)
        else:
            return KFold(n_splits=self.n_folds, shuffle=self.shuffle_kfold, random_state=self.random_state)


def is_constant(s, dropna=True) -> bool:
    if dropna:
        s = s[pd.notna(s)]
    a = s.to_numpy()
    return True if a.size == 0 else bool((a[0] == a).all())
