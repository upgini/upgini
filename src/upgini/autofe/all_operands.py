from typing import Dict

from upgini.autofe.binary import Add, Divide, Max, Min, Multiply, Sim, Subtract
from upgini.autofe.date import (
    DateDiff,
    DateDiffType2,
    DateListDiff,
    DateListDiffBounded,
    DatePercentile,
    DatePercentileMethod2,
)
from upgini.autofe.groupby import GroupByThenAgg, GroupByThenRank
from upgini.autofe.operand import Operand
from upgini.autofe.unary import Abs, Floor, Freq, Log, Residual, Norm, Sigmoid, Sqrt, Square
from upgini.autofe.vector import Mean, Sum

ALL_OPERANDS: Dict[str, Operand] = {
    op.name: op
    for op in [
        Freq(),
        Mean(),
        Sum(),
        Abs(),
        Log(),
        Sqrt(),
        Square(),
        Sigmoid(),
        Floor(),
        Residual(),
        Min(),
        Max(),
        Add(),
        Subtract(),
        Multiply(),
        Divide(),
        GroupByThenAgg(name="GroupByThenMin", agg="min"),
        GroupByThenAgg(name="GroupByThenMax", agg="max"),
        GroupByThenAgg(name="GroupByThenMean", agg="mean"),
        GroupByThenAgg(name="GroupByThenMedian", agg="median"),
        GroupByThenAgg(name="GroupByThenStd", output_type="float", agg="std"),
        GroupByThenRank(),
        Operand(name="Combine", has_symmetry_importance=True, output_type="object", is_categorical=True),
        Operand(name="CombineThenFreq", has_symmetry_importance=True, output_type="float"),
        Operand(name="GroupByThenNUnique", output_type="int", is_vectorizable=True, is_grouping=True),
        Operand(name="GroupByThenFreq", output_type="float", is_grouping=True),
        Sim(),
        DateDiff(),
        DateDiffType2(),
        DateListDiff(aggregation="min"),
        DateListDiff(aggregation="max"),
        DateListDiff(aggregation="mean"),
        DateListDiff(aggregation="nunique"),
        DateListDiffBounded(diff_unit="Y", aggregation="count", lower_bound=0, upper_bound=18),
        DateListDiffBounded(diff_unit="Y", aggregation="count", lower_bound=18, upper_bound=23),
        DateListDiffBounded(diff_unit="Y", aggregation="count", lower_bound=23, upper_bound=30),
        DateListDiffBounded(diff_unit="Y", aggregation="count", lower_bound=30, upper_bound=45),
        DateListDiffBounded(diff_unit="Y", aggregation="count", lower_bound=45, upper_bound=60),
        DateListDiffBounded(diff_unit="Y", aggregation="count", lower_bound=60),
        DatePercentile(),
        DatePercentileMethod2(),
        Norm(),
    ]
}


def find_op(name):
    return ALL_OPERANDS.get(name)
