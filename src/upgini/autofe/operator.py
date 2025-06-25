import abc
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, PrivateAttr


class OperatorRegistry(type(BaseModel)):
    _registry = {}
    _parametrized_registry = []

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        # Only register if it's a concrete class that inherits from Operand
        base_classes = [b for b in bases]
        base_names = {b.__name__ for b in bases}
        while base_classes:
            base = base_classes.pop()
            base_names.update(b.__name__ for b in base.__bases__)
            base_classes.extend(base.__bases__)

        if "Operator" in base_names:
            # Track parametrized operands separately
            if "ParametrizedOperator" in base_names:
                cls._parametrized_registry.append(new_class)
            else:
                try:
                    instance = new_class()
                    cls._registry[instance.name] = new_class
                    if instance.alias:
                        cls._registry[instance.alias] = new_class
                except Exception:
                    pass
        return new_class

    @classmethod
    def get_operator(cls, name: str) -> Optional["Operator"]:
        # First try to resolve as a parametrized operand formula
        for operator_cls in cls._parametrized_registry:
            resolved = operator_cls.from_formula(name)
            if resolved is not None:
                return resolved
        # Fall back to direct registry lookup
        non_parametrized = cls._registry.get(name)
        if non_parametrized is not None:
            return non_parametrized()
        return None


class Operator(BaseModel, metaclass=OperatorRegistry):
    name: Optional[str] = None
    alias: Optional[str] = None
    is_unary: bool = False
    is_symmetrical: bool = False
    has_symmetry_importance: bool = False
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    is_categorical: bool = False
    is_vectorizable: bool = False
    is_grouping: bool = False
    is_binary: bool = False
    is_vector: bool = False
    is_distribution_dependent: bool = False
    params: Optional[Dict[str, str]] = None

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    def set_params(self, params: Dict[str, str]):
        self.params = params
        return self

    def get_params(self) -> Dict[str, Optional[str]]:
        res = {"alias": self.alias}
        res.update(self.params or {})
        return res

    def to_formula(self) -> str:
        return self.name

    def get_hash_component(self) -> str:
        return self.to_formula()

    def set_logger(self, logger: logging.Logger):
        self._logger = logger
        return self

    def delete_data(self):
        pass


class ParametrizedOperator(Operator, abc.ABC):

    @abc.abstractmethod
    def to_formula(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def from_formula(cls, formula: str) -> Optional["Operator"]:
        pass


MAIN_COLUMN = "main_column"


class PandasOperator(Operator, abc.ABC):
    def calculate(self, **kwargs) -> pd.Series:
        if self.is_unary:
            return self.calculate_unary(kwargs["data"])
        elif self.is_binary or self.is_grouping:
            return self.calculate_binary(kwargs["left"], kwargs["right"])
        else:
            return self.calculate_vector(kwargs["data"])

    def calculate_unary(self, data: pd.Series) -> pd.Series:
        pass

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        pass

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        pass

    def calculate_group(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not self.is_vectorizable:
            raise RuntimeError(f"Cannot apply calculate_group: operator {self.name} is not vectorizable")
        else:
            raise RuntimeError(f"Unimplemented calculate_group for operator {self.name}")

    def _loc(self, df_to, df_from):
        df_from.loc[np.nan] = np.nan
        return df_to.fillna(np.nan).apply(lambda x: df_from.loc[x])

    def _round_value(
        self, value: Union[pd.Series, pd.DataFrame], precision: Optional[int] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(value, pd.DataFrame):
            return value.apply(self._round_value, axis=1)

        if np.issubdtype(value.dtype, np.floating):
            precision = precision or np.finfo(value.dtype).precision
            return np.trunc(value * 10**precision) / (10**precision)
        else:
            return value


class VectorizableMixin(Operator):
    group_index: int = 1

    def validate_calculation(self, input_columns: List[str], **kwargs) -> Tuple[str, List[str]]:
        if not kwargs.get(MAIN_COLUMN):
            raise ValueError(f"Expected argument {MAIN_COLUMN} for grouping operator {self.name} not found")
        group_column = kwargs[MAIN_COLUMN]
        value_columns = [col for col in input_columns if col != group_column]

        return group_column, value_columns
