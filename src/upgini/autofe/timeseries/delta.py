import pandas as pd
from typing import Dict, Optional

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase
from upgini.autofe.timeseries.lag import Lag


class Delta(TimeSeriesBase, ParametrizedOperator):
    delta_size: int
    delta_unit: str = "D"

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

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "delta_size": self.delta_size,
                "delta_unit": self.delta_unit,
                "offset_size": self.offset_size,
                "offset_unit": self.offset_unit,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        lag0 = Lag(lag_size=0, lag_unit=self.delta_unit)
        lag = Lag(lag_size=self.delta_size, lag_unit=self.delta_unit)
        return lag0._aggregate(ts) - lag._aggregate(ts)


class Delta2(TimeSeriesBase, ParametrizedOperator):
    delta_size: int
    delta_unit: str = "D"

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

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "delta_size": self.delta_size,
                "delta_unit": self.delta_unit,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        # Calculate first delta
        delta1 = Delta(delta_size=self.delta_size, delta_unit=self.delta_unit)
        first_delta = delta1._aggregate(ts)

        # Calculate delta of delta (second derivative)
        return delta1._aggregate(first_delta)
