import logging
from logging import Logger
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_object_dtype, is_string_dtype

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
        one_hot_encoded_features = []
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
            elif most_frequent_percent >= 0.99:
                if self.is_one_hot_encoded(column):
                    one_hot_encoded_features.append(f)
                else:
                    empty_or_constant_features.append(f)

        if one_hot_encoded_features:
            msg = bundle.get("one_hot_encoded_features").format(one_hot_encoded_features)
            warnings.append(msg)

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

    @staticmethod
    def is_one_hot_encoded(series: pd.Series) -> bool:
        try:
            # All rows should be the same type
            if series.apply(lambda x: type(x)).nunique() != 1:
                return False

            # First, handle string representations of True/False
            series_copy = series.copy()
            if series_copy.dtype == "object" or series_copy.dtype == "string":
                # Convert string representations of boolean values to numeric
                series_copy = series_copy.astype(str).str.strip().str.lower()
                series_copy = series_copy.replace({"true": "1", "false": "0"})

            # Column contains only 0 and 1 (as strings or numbers or booleans)
            series_copy = series_copy.astype(float)
            if set(series_copy.unique()) != {0.0, 1.0}:
                return False

            series_copy = series_copy.astype(int)

            # Column doesn't contain any NaN, np.NaN, space, null, etc.
            if not (series_copy.isin([0, 1])).all():
                return False

            vc = series_copy.value_counts()
            # Column should contain both 0 and 1
            if len(vc) != 2:
                return False

            # Minority class is 1
            if vc[1] >= vc[0]:
                return False

            return True
        except ValueError:
            return False
