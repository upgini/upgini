from upgini.autofe.operand import OperandRegistry
from upgini.autofe.binary import *  # noqa
from upgini.autofe.unary import *  # noqa
from upgini.autofe.groupby import *  # noqa
from upgini.autofe.vector import *  # noqa
from upgini.autofe.date import *  # noqa


def find_op(name):
    return OperandRegistry.get_operand(name)
