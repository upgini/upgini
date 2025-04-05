from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase


class TrendCoefficient(TimeSeriesBase, ParametrizedOperator):
    name: str = "trend_coef"
    step_size: int = 1
    step_unit: str = "D"

    def to_formula(self) -> str:
        base_formula = "trend_coef"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["TrendCoefficient"]:
        # Base regex for TrendCoefficient class
        base_regex = r"trend_coef"

        # Parse offset first
        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        # Basic pattern (no offset)
        if remaining_formula == "trend_coef":
            params = {}
            if offset_params:
                params.update(offset_params)
            return cls(**params)

        return None

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "step_size": self.step_size,
                "step_unit": self.step_unit,
                "offset_size": self.offset_size,
                "offset_unit": self.offset_unit,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(self._trend_coef).iloc[:, [-1]].fillna(0)

    def _trend_coef(self, x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        return_series = isinstance(x, pd.Series)
        x = pd.DataFrame(x)
        resampled = (
            x.iloc[:, -1].resample(f"{self.step_size}{self.step_unit}").fillna(method="ffill").fillna(method="bfill")
        )
        idx = np.arange(len(resampled))
        try:
            coeffs = np.polyfit(idx, resampled, 1)
            x.iloc[:, -1] = coeffs[0]
        except np.linalg.LinAlgError:
            x.iloc[:, -1] = 0
        return x.iloc[:, -1] if return_series else x
