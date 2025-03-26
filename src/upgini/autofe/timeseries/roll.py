import pandas as pd
from typing import Dict, Optional

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase

# Roll aggregation functions
roll_aggregations = {
    "norm_mean": lambda x: x[-1] / x.mean(),
    "q25": lambda x: x.quantile(0.25),
    "q75": lambda x: x.quantile(0.75),
    "iqr": lambda x: x.quantile(0.75) - x.quantile(0.25),
}

try:
    from pydantic import field_validator as validator  # V2
except ImportError:
    from pydantic import validator  # V1


class Roll(TimeSeriesBase, ParametrizedOperator):
    aggregation: str
    window_size: int = 1
    window_unit: str = "D"

    @validator("window_unit")
    @classmethod
    def validate_window_unit(cls, v: str) -> str:
        try:
            pd.tseries.frequencies.to_offset(v)
            return v
        except ValueError:
            raise ValueError(
                f"Invalid window_unit: {v}. Must be a valid pandas frequency string (e.g. 'D', 'H', 'T', etc)"
            )

    def to_formula(self) -> str:
        roll_component = f"roll_{self.window_size}{self.window_unit}"
        if self.offset_size > 0:
            roll_component += f"_offset_{self.offset_size}{self.offset_unit}"
        return f"{roll_component}_{self.aggregation}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Roll"]:
        import re

        # Try matching pattern with offset first
        pattern_with_offset = r"^roll_(\d+)([a-zA-Z])_offset_(\d+)([a-zA-Z])_(\w+)$"
        match_with_offset = re.match(pattern_with_offset, formula)

        if match_with_offset:
            window_size = int(match_with_offset.group(1))
            window_unit = match_with_offset.group(2)
            offset_size = int(match_with_offset.group(3))
            offset_unit = match_with_offset.group(4)
            aggregation = match_with_offset.group(5)

            return cls(
                window_size=window_size,
                window_unit=window_unit,
                offset_size=offset_size,
                offset_unit=offset_unit,
                aggregation=aggregation,
            )

        # If no offset pattern found, try basic pattern
        pattern = r"^roll_(\d+)([a-zA-Z])_(\w+)$"
        match = re.match(pattern, formula)

        if not match:
            return None

        window_size = int(match.group(1))
        window_unit = match.group(2)
        aggregation = match.group(3)

        return cls(window_size=window_size, window_unit=window_unit, aggregation=aggregation)

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
