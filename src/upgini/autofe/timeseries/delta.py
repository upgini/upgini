import pandas as pd
from typing import Dict, Optional

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase
from upgini.autofe.timeseries.numpy_kernels import apply_grouped_kernel, delta2_values, delta_values


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
        delta_size = self.delta_size
        delta_unit = self.delta_unit
        return apply_grouped_kernel(
            ts, lambda times, values: delta_values(times, values, delta_size, delta_unit)
        )


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
        delta_size = self.delta_size
        delta_unit = self.delta_unit
        return apply_grouped_kernel(
            ts, lambda times, values: delta2_values(times, values, delta_size, delta_unit)
        )
