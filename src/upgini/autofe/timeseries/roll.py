import pandas as pd
from typing import Dict, Optional

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase
from upgini.autofe.utils import pydantic_validator

# Roll aggregation functions
roll_aggregations = {
    "norm_mean": lambda x: x[-1] / x.mean(),
    "q25": lambda x: x.quantile(0.25),
    "q75": lambda x: x.quantile(0.75),
    "iqr": lambda x: x.quantile(0.75) - x.quantile(0.25),
}


class Roll(TimeSeriesBase, ParametrizedOperator):
    aggregation: str
    window_size: int = 1
    window_unit: str = "D"

    @pydantic_validator("window_unit")
    def validate_window_unit(cls, v: str) -> str:
        try:
            pd.tseries.frequencies.to_offset(v)
            return v
        except ValueError:
            raise ValueError(
                f"Invalid window_unit: {v}. Must be a valid pandas frequency string (e.g. 'D', 'H', 'T', etc)"
            )

    def to_formula(self) -> str:
        # First add window size and unit, then add aggregation, then add offset
        base_formula = f"roll_{self.window_size}{self.window_unit}"
        formula_with_agg = f"{base_formula}_{self.aggregation}"
        return self._add_offset_to_formula(formula_with_agg)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Roll"]:
        import re

        # Base regex for Roll class (with aggregation)
        base_regex = r"roll_(\d+)([a-zA-Z])_(\w+)"

        # Parse offset first - this removes the offset part if present
        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        # Parse the window part and aggregation
        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        window_size = int(match.group(1))
        window_unit = match.group(2)
        aggregation = match.group(3)

        # Create instance with appropriate parameters
        params = {
            "window_size": window_size,
            "window_unit": window_unit,
            "aggregation": aggregation,
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "window_size": self.window_size,
                "window_unit": self.window_unit,
                "aggregation": self.aggregation,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.rolling(f"{self.window_size}{self.window_unit}", min_periods=1).agg(
            roll_aggregations.get(self.aggregation, self.aggregation)
        )
