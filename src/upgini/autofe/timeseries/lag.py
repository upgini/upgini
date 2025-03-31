import numpy as np
import pandas as pd
from typing import Dict, Optional

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase


class Lag(TimeSeriesBase, ParametrizedOperator):
    lag_size: int
    lag_unit: str = "D"

    def to_formula(self) -> str:
        base_formula = f"lag_{self.lag_size}{self.lag_unit}"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Lag"]:
        # Base regex for Lag class
        base_regex = r"lag_(\d+)([a-zA-Z])"

        # Parse offset first
        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        # Now parse the lag part
        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        lag_size = int(match.group(1))
        lag_unit = match.group(2)

        # Create instance with appropriate parameters
        params = {
            "lag_size": lag_size,
            "lag_unit": lag_unit,
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

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
