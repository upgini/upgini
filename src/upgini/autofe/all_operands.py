from typing import Dict
from upgini.autofe.groupby import GroupByThenAgg, GroupByThenRank
from upgini.autofe.operand import Operand
from upgini.autofe.unary import Abs, Log, Residual, Sqrt, Square, Sigmoid, Floor, Freq
from upgini.autofe.binary import Min, Max, Add, Subtract, Multiply, Divide, Sim
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
    ]
}


def find_op(name):
    return ALL_OPERANDS.get(name)
