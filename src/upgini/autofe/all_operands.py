from upgini.autofe.operator import OperatorRegistry
from upgini.autofe.unary import *  # noqa
from upgini.autofe.binary import *  # noqa
from upgini.autofe.groupby import *  # noqa
from upgini.autofe.date import *  # noqa
from upgini.autofe.vector import *  # noqa


def find_op(name):
    return OperatorRegistry.get_operator(name)
