import logging
from logging import Logger
from typing import List, Optional

import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype


class FeaturesValidator:
    def __init__(self, logger: Optional[Logger] = None):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")

    def validate(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        empty_or_constant_features = []
        high_cardinality_features = []
        count = len(df)

        for f in features:
            value_counts = df[f].value_counts(dropna=False, normalize=True)
            most_frequent_percent = value_counts.iloc[0]
            if most_frequent_percent >= 0.99:
                empty_or_constant_features.append(f)
                continue

            if (is_string_dtype(df[f]) or is_integer_dtype(df[f])) and df[f].nunique() / count >= 0.9:
                high_cardinality_features.append(f)
                continue

        if empty_or_constant_features:
            msg = (
                f"Columns {empty_or_constant_features} has value with frequency more than 99% "
                "and has been droped from X"
            )
            print(msg)
            self.logger.warning(msg)

        if high_cardinality_features:
            msg = (
                f"Columns {high_cardinality_features} has high cardinality (>90% unique values) "
                "and has been droped from X"
            )
            print(msg)
            self.logger.warning(msg)

        return empty_or_constant_features + high_cardinality_features
