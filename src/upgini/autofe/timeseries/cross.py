import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from upgini.autofe.all_operators import find_op
from upgini.autofe.operator import PandasOperator, ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase
from upgini.autofe.utils import pydantic_validator


class CrossSeriesInteraction(TimeSeriesBase, ParametrizedOperator):
    base_name: str = "cross"
    interaction_op: PandasOperator
    descriptor_indices: List[int] = []
    left_descriptor: List[str] = []
    right_descriptor: List[str] = []

    @pydantic_validator("descriptor_indices", mode="before")
    def validate_descriptor_indices(cls, v):
        if isinstance(v, str):
            v = json.loads(v)
        if not v:
            raise ValueError("descriptor_indices cannot be empty")
        return v

    @pydantic_validator("left_descriptor", "right_descriptor", mode="before")
    def parse_descriptors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @pydantic_validator("interaction_op", mode="before")
    def validate_interaction_op(cls, v):
        if isinstance(v, str):
            return find_op(v)
        return v

    def __init__(self, **data):
        super().__init__(**data)
        indices = self.descriptor_indices
        left = self.left_descriptor
        right = self.right_descriptor

        if len(left) != len(indices):
            raise ValueError(
                f"left_descriptor length ({len(left)}) " f"must match descriptor_indices length ({len(indices)})"
            )

        if len(right) != len(indices):
            raise ValueError(
                f"right_descriptor length ({len(right)}) " f"must match descriptor_indices length ({len(indices)})"
            )

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

        # Include default values to pass validation
        params = {
            "interaction_op": op,
            "descriptor_indices": [0],  # Default index
            "left_descriptor": ["default"],  # Default left descriptor
            "right_descriptor": ["default"],  # Default right descriptor
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "interaction_op": self._get_interaction_op_name(),
                "descriptor_indices": json.dumps(self.descriptor_indices),
                "left_descriptor": json.dumps(self.left_descriptor),
                "right_descriptor": json.dumps(self.right_descriptor),
            }
        )
        return res

    def get_hash_component(self) -> str:
        inner_components = [
            self.to_formula(),
            "_".join(str(i) for i in self.descriptor_indices),
            "_".join(self.left_descriptor),
            "_".join(self.right_descriptor),
        ]
        return "_".join(inner_components)

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
        mask = np.logical_and.reduce([data[i].astype(str) == v for i, v in zip(self.descriptor_indices, descriptor)])
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
