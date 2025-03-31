from upgini.autofe.operator import OperatorRegistry


def test_get_operands_from_registry():
    # Test frequency operand
    freq = OperatorRegistry.get_operator("freq")
    assert freq is not None
    assert freq.name == "freq"
    assert freq.__class__.__name__ == "Freq"

    # Test parametrized group by operand
    parsed = OperatorRegistry.get_operator("GroupByThenMin")
    assert parsed is not None
    assert parsed.agg == "Min"
    assert parsed.is_grouping is True
    assert parsed.__class__.__name__ == "GroupByThenAgg"

    # Test parametrized date diff operand
    parsed = OperatorRegistry.get_operator("date_diff_Y_18_23_count")
    assert parsed is not None
    assert parsed.diff_unit == "Y"
    assert parsed.lower_bound == 18
    assert parsed.upper_bound == 23
    assert parsed.aggregation == "count"
    assert parsed.__class__.__name__ == "DateListDiffBounded"

    # Test parametrized roll operand
    parsed = OperatorRegistry.get_operator("roll_3d_mean")
    assert parsed is not None
    assert parsed.window_size == 3
    assert parsed.window_unit == "d"
    assert parsed.aggregation == "mean"
    assert parsed.__class__.__name__ == "Roll"

    parsed_add_name = OperatorRegistry.get_operator("+")
    parsed_add_alias = OperatorRegistry.get_operator("add")
    assert parsed_add_name.__class__.__name__ == parsed_add_alias.__class__.__name__

    # Test non-existent operand
    assert OperatorRegistry.get_operator("not_an_operand") is None
