import logging
from logging import Logger
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)

from upgini.resource_bundle import bundle


class FeaturesValidator:
    def __init__(self, logger: Optional[Logger] = None):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")

    def validate(
        self,
        df: pd.DataFrame,
        features: List[str],
        features_for_generate: Optional[List[str]] = None,
        columns_renaming: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], List[str]]:
        empty_or_constant_features = []
        high_cardinality_features = []
        warnings = []

        for f in features:
            column = df[f]
            if is_object_dtype(column):
                column = column.astype("string")
            value_counts = column.value_counts(dropna=False, normalize=True)
            most_frequent_percent = value_counts.iloc[0]

            if len(value_counts) == 1:
                empty_or_constant_features.append(f)
            elif most_frequent_percent >= 0.99 and len(value_counts) != 2 and not self.__is_integer(column):
                empty_or_constant_features.append(f)

        columns_renaming = columns_renaming or {}

        if empty_or_constant_features:
            msg = bundle.get("empty_or_contant_features").format(
                [columns_renaming.get(f, f) for f in empty_or_constant_features]
            )
            warnings.append(msg)

        high_cardinality_features = self.find_high_cardinality(df[features])
        if features_for_generate:
            high_cardinality_features = [
                f for f in high_cardinality_features if columns_renaming.get(f, f) not in features_for_generate
            ]
        if high_cardinality_features:
            msg = bundle.get("high_cardinality_features").format(
                [columns_renaming.get(f, f) for f in high_cardinality_features]
            )
            warnings.append(msg)

        return (empty_or_constant_features + high_cardinality_features, warnings)

    @staticmethod
    def find_high_cardinality(df: pd.DataFrame) -> List[str]:
        # Remove high cardinality columns
        row_count = df.shape[0]
        if row_count < 100:  # For tests with small datasets
            return []
        return [
            i
            for i in df
            if (is_object_dtype(df[i]) or is_string_dtype(df[i]) or FeaturesValidator.__is_integer(df[i]))
            and (df[i].nunique(dropna=False) / row_count >= 0.85)
        ]

    @staticmethod
    def __is_integer(series: pd.Series) -> bool:
        return (
            is_integer_dtype(series)
            or series.dropna()
            .apply(
                lambda f: (float.is_integer(f) and abs(f) < np.iinfo(np.int64).max) if isinstance(f, float) else False
            )
            .all()
        )

    @staticmethod
    def find_constant_features(df: pd.DataFrame) -> List[str]:
        return [i for i in df if df[i].nunique() <= 1]


def is_effective_int(column: pd.Series) -> bool:
    """
    Return True if the series is effectively integer:
    - integer dtype; or
    - float dtype with all non-null values being whole numbers within int64 range.
    """
    if is_integer_dtype(column):
        return True
    if is_float_dtype(column):
        non_null_values = column.dropna()
        if non_null_values.empty:
            return False
        within_range_mask = non_null_values.abs() < np.iinfo(np.int64).max
        is_integer_mask = non_null_values.apply(lambda v: float(v).is_integer())
        return bool((within_range_mask & is_integer_mask).all())
    return False
