from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from upgini.autofe.all_operands import find_op
from upgini.autofe.operator import PandasOperator, ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase


class CrossSeriesInteraction(TimeSeriesBase, ParametrizedOperator):
    base_name: str = "cross"
    interaction_op: PandasOperator
    descriptor_indices: List[int] = []
    left_descriptor: List[str] = []
    right_descriptor: List[str] = []

    def to_formula(self) -> str:
        base_formula = f"{self.base_name}_{self._get_interaction_op_name()}"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["CrossSeriesInteraction"]:
        base_regex = r"cross_(.+)"

        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        # Extract the operator formula
        op_formula = match.group(1)

        op = find_op(op_formula)
        if op is None or not op.is_binary:
            return None

        params = {"op": op}

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def get_params(self) -> Dict[str, str | None]:
        res = super().get_params()
        res.update(
            {
                "interaction_op": self._get_interaction_op_name(),
                "descriptor_indices": self.descriptor_indices,
                "left_descriptor": self.left_descriptor,
                "right_descriptor": self.right_descriptor,
            }
        )
        return res

    def _get_interaction_op_name(self) -> str:
        return self.interaction_op.alias or self.interaction_op.to_formula()

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        left_mask = self._get_mask(data, self.left_descriptor)
        left = self._extract_series(data, left_mask)

        right_mask = self._get_mask(data, self.right_descriptor)
        right = self._extract_series(data, right_mask)

        interaction: pd.Series = self.interaction_op.calculate_binary(left, right)
        interaction = interaction.reindex(self._get_index(data))
        res = pd.Series(np.nan, index=data[-1].index, name=data[-1].name)
        res.loc[left_mask] = interaction[left_mask].values
        res.loc[right_mask] = interaction[right_mask].values
        return res

    def _get_mask(self, data: List[pd.Series], descriptor: List[str]) -> pd.Series:
        mask = np.logical_and.reduce([data[i] == v for i, v in zip(self.descriptor_indices, descriptor)])
        return mask

    def _extract_series(self, data: List[pd.Series], mask: pd.Series) -> pd.Series:
        masked_data = [d[mask] for d in data]
        shifted = super().calculate_vector(masked_data)
        shifted.index = self._get_index(masked_data)
        return shifted

    def _get_index(self, data: List[pd.Series]) -> pd.Series:
        index = [d for i, d in enumerate(data[:-1]) if i not in self.descriptor_indices]
        return index if len(index) > 1 else index[0]

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(lambda x: x).iloc[:, [-1]]
