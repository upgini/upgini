from upgini.autofe.operator import OperatorRegistry
from upgini.autofe.timeseries import Roll
from upgini.autofe.utils import pydantic_parse_method


def test_parametrized_operator_registry_roundtrip():
    roll_from_registry = OperatorRegistry.get_operator("roll_3d_mean")
    assert roll_from_registry is not None

    roll_dict = roll_from_registry.get_params()
    parsed_roll = pydantic_parse_method(Roll)(roll_dict)

    assert parsed_roll.window_size == 3
    assert parsed_roll.window_unit == "d"
    assert parsed_roll.aggregation == "mean"
    assert parsed_roll.to_formula() == "roll_3d_mean"

    formula = parsed_roll.to_formula()

    # Verify we can get it back from the registry
    retrieved_roll = OperatorRegistry.get_operator(formula)
    assert retrieved_roll is not None
    assert retrieved_roll.to_formula() == formula
