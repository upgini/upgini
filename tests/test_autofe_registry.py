from upgini.autofe.operand import OperandRegistry


def test_get_operands_from_registry():
    # Test frequency operand
    freq = OperandRegistry.get_operand("freq")
    assert freq is not None
    assert freq.name == "freq"
    assert freq.__class__.__name__ == "Freq"

    # Test parametrized group by operand
    parsed = OperandRegistry.get_operand("GroupByThenMin")
    assert parsed is not None
    assert parsed.agg == "min"
    assert parsed.is_grouping is True
    assert parsed.__class__.__name__ == "GroupByThenAgg"

    # Test parametrized date diff operand
    parsed = OperandRegistry.get_operand("date_diff_Y_18_23_count")
    assert parsed is not None
    assert parsed.diff_unit == "Y"
    assert parsed.lower_bound == 18
    assert parsed.upper_bound == 23
    assert parsed.aggregation == "count"
    assert parsed.__class__.__name__ == "DateListDiffBounded"

    # Test parametrized roll operand
    parsed = OperandRegistry.get_operand("roll_3d_mean")
    assert parsed is not None
    assert parsed.window_size == 3
    assert parsed.window_unit == "d"
    assert parsed.aggregation == "mean"
    assert parsed.__class__.__name__ == "Roll"

    # Test non-existent operand
    assert OperandRegistry.get_operand("not_an_operand") is None
