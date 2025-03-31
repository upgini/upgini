import pandas as pd
from typing import Dict, Optional, Union

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase
from upgini.autofe.timeseries.lag import Lag


class DeltaBase(TimeSeriesBase):
    delta_size: int
    delta_unit: str = "D"

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "delta_size": self.delta_size,
                "delta_unit": self.delta_unit,
            }
        )
        return res

    def _calculate_delta(self, x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        return_series = isinstance(x, pd.Series)
        x = pd.DataFrame(x)
        lag = Lag(lag_size=self.delta_size, lag_unit=self.delta_unit)
        x.iloc[:, -1] = x.iloc[:, -1] - lag._aggregate(x.iloc[:, -1])
        return x.iloc[:, -1] if return_series else x


class Delta(DeltaBase, ParametrizedOperator):
    def to_formula(self) -> str:
        base_formula = f"delta_{self.delta_size}{self.delta_unit}"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Delta"]:
        # Base regex for Delta class
        base_regex = r"delta_(\d+)([a-zA-Z])"

        # Parse offset first
        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        # Now parse the delta part
        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        delta_size = int(match.group(1))
        delta_unit = match.group(2)

        # Create instance with appropriate parameters
        params = {
            "delta_size": delta_size,
            "delta_unit": delta_unit,
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(self._calculate_delta).iloc[:, [-1]]


class Delta2(DeltaBase, ParametrizedOperator):
    def to_formula(self) -> str:
        base_formula = f"delta2_{self.delta_size}{self.delta_unit}"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Delta2"]:
        # Base regex for Delta2 class
        base_regex = r"delta2_(\d+)([a-zA-Z])"

        # Parse offset first
        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        # Now parse the delta part
        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        delta_size = int(match.group(1))
        delta_unit = match.group(2)

        # Create instance with appropriate parameters
        params = {
            "delta_size": delta_size,
            "delta_unit": delta_unit,
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(self._calculate_delta2).iloc[:, [-1]]

    def _calculate_delta2(self, x):
        # Calculate first delta
        first_delta = self._calculate_delta(x)

        # Calculate delta of delta (second derivative)
        return self._calculate_delta(first_delta)
