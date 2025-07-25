import hashlib
import itertools
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pandas._typing import DtypeObj

from upgini.autofe.all_operators import find_op
from upgini.autofe.operator import Operator, PandasOperator
from upgini.autofe.timeseries.base import TimeSeriesBase
from upgini.autofe.utils import pydantic_dump_method, pydantic_parse_method


class Column:
    def __init__(self, name: str, data: Optional[pd.Series] = None, calculate_all=False):
        self.name = name
        self.data = data
        self.calculate_all = calculate_all

    def set_op_params(self, params: Dict[str, str], **kwargs) -> "Column":
        return self

    def get_op_params(self, **kwargs):
        return dict()

    def rename_columns(self, mapping: Dict[str, str]) -> "Column":
        self.name = mapping.get(self.name) or self.name
        return self

    def delete_data(self):
        self.data = None

    def get_column_nodes(self) -> List["Column"]:
        return [self]

    def get_columns(self, unhash=False, **kwargs):
        name = self.name
        return [self._unhash(name) if unhash else name]

    def get_display_name(self, cache: bool = True, shorten: bool = False, **kwargs) -> str:
        return self.get_columns(**kwargs)[0]

    def _unhash(self, feature_name: str) -> str:
        last_component_idx = feature_name.rfind("_")
        if not feature_name.startswith("f_"):
            return feature_name  # etalon feature
        elif last_component_idx == 1:
            return feature_name[2:]  # fully hashed name, cannot unhash
        else:
            return feature_name[2:last_component_idx]

    @property
    def children(self) -> List[Union["Feature", "Column"]]:
        return []

    def infer_type(self, data: pd.DataFrame) -> DtypeObj:
        return data[self.name].dtype

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.data = data[self.name]
        return self.data

    def to_formula(self, **kwargs) -> str:
        return str(self.get_columns(**kwargs)[0])

    def to_pretty_formula(self) -> str:
        return self.to_formula()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Column):
            return False
        else:
            return self.name == value.name and self.calculate_all == value.calculate_all


class Feature:
    def __init__(
        self,
        op: Operator,
        children: List[Union[Column, "Feature"]],
        data: Optional[pd.DataFrame] = None,
        display_index: Optional[str] = None,
        cached_display_name: Optional[str] = None,
        alias: Optional[str] = None,
    ):
        self.op = op
        self.children = children
        self.data = data
        self.display_index = display_index
        self.cached_display_name = cached_display_name
        self.alias = alias

    def set_op_params(self, params: Optional[Dict[str, str]], **kwargs) -> "Feature":
        obj_dict = pydantic_dump_method(self.op)().copy()
        obj_dict.update(params or {})
        self.op = pydantic_parse_method(self.op.__class__)(obj_dict)
        self.op.set_params(params)

        for child in self.children:
            child_params = {
                k[len(child.get_display_name(**kwargs)) + 1 :]: v
                for k, v in params.items()
                if k.startswith(child.get_display_name(**kwargs))
            }
            if not child_params:
                child_params = params
            child.set_op_params(child_params, **kwargs)
        return self

    def get_op_params(self, **kwargs) -> Dict[str, str]:
        return {
            k: str(v)
            for k, v in dict(
                (
                    (f"{child.get_display_name(**kwargs)}_{k}", v)
                    for child in self.children
                    for k, v in child.get_op_params(**kwargs).items()
                ),
                **(self.op.get_params() or {}),
            ).items()
            if v is not None
        }

    def get_hash(self) -> str:
        return hashlib.sha256(
            "_".join(
                [self.op.get_hash_component()]
                + [ch.op.get_hash_component() for ch in self.children if isinstance(ch, Feature)]
                + [ch.get_display_name() for ch in self.children]
            ).encode("utf-8")
        ).hexdigest()[:10]

    def set_alias(self, alias: str) -> "Feature":
        self.alias = alias
        return self

    def get_all_operand_names(self) -> Set[str]:
        return {self.op.to_formula()}.union(
            {n for f in self.children if isinstance(f, Feature) for n in f.get_all_operand_names()}
        )

    def rename_columns(self, mapping: Dict[str, str]) -> "Feature":
        for child in self.children:
            child.rename_columns(mapping)
        self.cached_display_name = None
        return self

    def get_column_nodes(self) -> List[Union[Column, "Feature"]]:
        res = []
        for child in self.children:
            res.extend(child.get_column_nodes())
        return res

    def get_columns(self, **kwargs) -> List[str]:
        column_list = []
        seen = set()
        for child in self.children:
            columns = child.get_columns(**kwargs)
            column_list.extend([f for f in columns if f not in seen])
            seen.update(columns)
        return column_list

    def delete_data(self):
        self.data = None
        self.op.delete_data()
        for child in self.children:
            child.delete_data()

    def get_op_display_name(self, use_alias: bool = True) -> str:
        return (self.op.alias or self.op.to_formula()).lower() if use_alias else self.op.to_formula()

    def get_display_name(self, cache: bool = True, shorten: bool = False, use_op_alias: bool = True, **kwargs) -> str:
        if self.cached_display_name is not None and cache:
            return self.cached_display_name

        should_stack_op = (
            not isinstance(self.children[-1], Column)
            if self.op.is_unary or isinstance(self.op, TimeSeriesBase)
            else False
        )
        components = []

        if self.alias:
            components.extend(["f_autofe", self.alias])
        elif should_stack_op:
            components.extend(
                [
                    self.children[-1].get_display_name(
                        cache=cache, shorten=shorten, use_op_alias=use_op_alias, **kwargs
                    ),
                    self.get_op_display_name(use_alias=use_op_alias),
                ]
            )
        elif shorten and not self.op.is_unary:
            components.extend(["f_autofe", self.get_op_display_name(use_alias=use_op_alias)])
        else:
            components = ["f_" + "_f_".join(self.get_columns(**kwargs))] + [
                "autofe",
                self.get_op_display_name(use_alias=use_op_alias),
            ]
        components.extend([str(self.display_index)] if self.display_index is not None else [])
        display_name = "_".join(components)

        if cache:
            self.cached_display_name = display_name
        return display_name

    def set_display_index(self, index) -> "Feature":
        self.display_index = index
        self.cached_display_name = None
        return self

    def infer_type(self, data: pd.DataFrame) -> Union[str, DtypeObj]:
        if self.op.output_type:
            return self.op.output_type
        else:
            # either a symmetrical operator or group by
            return self.children[0].infer_type(data)

    def calculate(self, data: pd.DataFrame, is_root=False) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(self.op, PandasOperator):
            if self.op.is_vector:
                ds = [child.calculate(data) for child in self.children]
                new_data = self.op.calculate(data=ds)
            else:
                d1 = self.children[0].calculate(data)
                d2 = None if len(self.children) < 2 else self.children[1].calculate(data)
                new_data = self.op.calculate(data=d1, left=d1, right=d2)
        else:
            raise NotImplementedError(f"Unrecognized operator {self.op.name}.")

        if (str(new_data.dtype) == "category") | (str(new_data.dtype) == "object"):
            pass
        else:
            new_data = new_data.replace([-np.inf, np.inf], np.nan)

        if is_root:
            self.data = new_data
        return new_data

    @staticmethod
    def check_xor(left: Union[Column, "Feature"], right: Union[Column, "Feature"]) -> bool:
        def _get_all_columns(feature: Union[Column, "Feature"]) -> List[str]:
            if isinstance(feature, Column):
                return [feature.name]
            else:
                res = []
                for child in feature.children:
                    res.extend(_get_all_columns(child))
                return res

        column1 = set(_get_all_columns(left))
        column2 = set(_get_all_columns(right))
        if len(column1 ^ column2) == 0:
            return False
        else:
            return True

    def to_formula(self, **kwargs) -> str:
        if self.op.name in ["+", "-", "*", "/"]:
            left = self.children[0].to_formula(**kwargs)
            right = self.children[1].to_formula(**kwargs)
            return f"({left}{self.op.to_formula()}{right})"
        else:
            result = [self.op.to_formula(), "("]
            for i in range(len(self.children)):
                string_i = self.children[i].to_formula(**kwargs)
                result.append(string_i)
                result.append(",")
            result.pop()
            result.append(")")
            return "".join(result)

    def to_pretty_formula(self) -> str:
        if self.op.name in ["+", "-", "*", "/"]:
            left = self.children[0].to_pretty_formula()
            right = self.children[1].to_pretty_formula()
            return f"{left} {self.op.to_formula()} {right}"
        else:
            result = [self.op.to_formula(), "("]
            for i in range(len(self.children)):
                string_i = self.children[i].to_pretty_formula()
                result.append(string_i)
                result.append(", ")
            result.pop()
            result.append(")")
            return "".join(result)

    @staticmethod
    def from_formula(string: str) -> Union[Column, "Feature"]:

        def is_trivial_char(c: str) -> bool:
            return c not in "()+-*/,"

        if string[-1] != ")":
            if all(is_trivial_char(c) for c in string):
                return Column(string)
            else:
                raise ValueError(
                    f"Unsupported column name: {string}. Column names should not have characters: "
                    "['(', ')', '+', '-', '*', '/', ',']"
                )

        def find_prev(string: str) -> int:
            if string[-1] != ")":
                return max([(0 if is_trivial_char(c) else i + 1) for i, c in enumerate(string)])
            level, pos = 0, -1
            for i in range(len(string) - 1, -1, -1):
                if string[i] == ")":
                    level += 1
                if string[i] == "(":
                    level -= 1
                if level == 0:
                    pos = i
                    break
            while (pos > 0) and is_trivial_char(string[pos - 1]):
                pos -= 1
            return pos

        p2 = find_prev(string[:-1])
        if string[p2 - 1] == "(":
            return Feature(find_op(string[: p2 - 1]), [Feature.from_formula(string[p2:-1])])
        p1 = find_prev(string[: p2 - 1])
        if string[0] == "(":
            op = find_op(string[p2 - 1])
            if op is None:
                raise ValueError(f"Unsupported operand: {string[p2 - 1]}")
            return Feature(
                op,
                [Feature.from_formula(string[p1 : p2 - 1]), Feature.from_formula(string[p2:-1])],
            )
        else:
            op = find_op(string[: p1 - 1])
            if op is not None:
                return Feature(
                    op,
                    [Feature.from_formula(string[p1 : p2 - 1]), Feature.from_formula(string[p2:-1])],
                )
            else:
                if string[p1 - 1] == "(":
                    raise ValueError(f"Unsupported operand: {string[: p1 - 1]}")
                base_features = [
                    Feature.from_formula(string[p2:-1]),
                    Feature.from_formula(string[p1 : p2 - 1]),
                ]
                while op is None:
                    p2 = p1
                    p1 = find_prev(string[: p1 - 1])
                    base_features.append(Feature.from_formula(string[p1 : p2 - 1]))
                    op = find_op(string[: p1 - 1])
                base_features.reverse()
                return Feature(op, base_features)

    def set_logger(self, logger: logging.Logger):
        self.op.set_logger(logger)
        for child in self.children:
            if hasattr(child, "set_logger"):
                child.set_logger(logger)
        return self


class FeatureGroup:
    def __init__(
        self, op: Operator, main_column: Optional[Union[Column, Feature]], children: List[Union[Column, Feature]]
    ):
        self.op = op
        self.main_column_node = main_column
        self.children = children
        self.data: Optional[pd.DataFrame] = None

    def get_columns(self, **kwargs) -> List[str]:
        column_list = []
        seen = set()
        for child in self.children:
            columns = child.get_columns(**kwargs)
            column_list.extend([f for f in columns if f not in seen])
            seen.update(columns)
        return column_list

    def get_display_names(self, **kwargs) -> List[str]:
        names = [f.get_display_name(**kwargs) for f in self.children]
        return names

    def calculate(self, data: pd.DataFrame, is_root=False) -> pd.DataFrame:
        if isinstance(self.op, PandasOperator):
            main_column = None if self.main_column_node is None else self.main_column_node.get_display_name()
            lower_order_children = []
            if self.main_column_node is not None:
                lower_order_children.append(self.main_column_node)
            lower_order_children.extend(
                ch for f in self.children for ch in f.children if ch.get_display_name() != main_column
            )
            lower_order_names = [ch.get_display_name() for ch in lower_order_children]
            child_data = pd.concat(
                [ch.calculate(data) for ch in lower_order_children],
                axis=1,
            )
            child_data.columns = lower_order_names

            new_data = self.op.calculate_group(child_data, main_column=main_column)
            new_data.rename(
                columns=dict(zip((n for n in lower_order_names if n != main_column), self.get_display_names())),
                inplace=True,
            )
        else:
            raise NotImplementedError(f"Unrecognized operator {self.op.name}.")

        new_data.replace([-np.inf, np.inf], np.nan, inplace=True)

        if is_root:
            self.data = new_data
        return new_data

    @staticmethod
    def make_groups(candidates: List[Feature]) -> List[Union[Feature, "FeatureGroup"]]:
        grouped_features = []

        def groupby_func(f: Feature) -> Tuple[Operator, Union[Column, Feature]]:
            return (f.op, f.children[0 if not f.op.is_vectorizable else f.op.group_index])

        for op_child, features in itertools.groupby(candidates, groupby_func):
            op, main_child = op_child
            feature_list = list(features)
            if op.is_vectorizable:
                if op.is_unary:
                    group = FeatureGroup(op, main_column=None, children=feature_list)
                else:
                    group = FeatureGroup(op, main_column=main_child, children=feature_list)
                grouped_features.append(group)
            else:
                grouped_features.extend(feature_list)
        return grouped_features

    def delete_data(self):
        self.data = None
        if self.main_column_node:
            self.main_column_node.delete_data()
        for child in self.children:
            child.delete_data()
