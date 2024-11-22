from upgini.autofe.operand import OperandRegistry


def find_op(name):
    return OperandRegistry.get_operand(name)
