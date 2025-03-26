import numpy as np
import pandas as pd
from typing import Dict, Optional

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase


class Lag(TimeSeriesBase, ParametrizedOperator):
    lag_size: int
    lag_unit: str = "D"

    def to_formula(self) -> str:
        lag_component = f"lag_{self.lag_size}{self.lag_unit}"
        if self.offset_size > 0:
            lag_component += f"_offset_{self.offset_size}{self.offset_unit}"
        return lag_component

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Lag"]:
        import re

        # Try matching pattern with offset first
        pattern_with_offset = r"^lag_(\d+)([a-zA-Z])_offset_(\d+)([a-zA-Z])$"
        match_with_offset = re.match(pattern_with_offset, formula)

        if match_with_offset:
            lag_size = int(match_with_offset.group(1))
            lag_unit = match_with_offset.group(2)
            offset_size = int(match_with_offset.group(3))
            offset_unit = match_with_offset.group(4)

            return cls(
                lag_size=lag_size,
                lag_unit=lag_unit,
                offset_size=offset_size,
                offset_unit=offset_unit,
            )

        # If no offset pattern found, try basic pattern
        pattern = r"^lag_(\d+)([a-zA-Z])$"
        match = re.match(pattern, formula)

        if not match:
            return None

        lag_size = int(match.group(1))
        lag_unit = match.group(2)

        return cls(lag_size=lag_size, lag_unit=lag_unit)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "lag_size": self.lag_size,
                "lag_unit": self.lag_unit,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        lag_window = self.lag_size + 1
        return ts.rolling(f"{lag_window}{self.lag_unit}", min_periods=1).agg(self._lag)

    def _lag(self, x):
        if x.index.min() > (x.index.max() - pd.Timedelta(self.lag_size, self.lag_unit)):
            return np.nan
        else:
            return x[0]
