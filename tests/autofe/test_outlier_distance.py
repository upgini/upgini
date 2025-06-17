from upgini.autofe.unary import OutlierDistance


def test_outlier_distance_formula_conversion():
    # Test with no class value
    op = OutlierDistance(centroid="[1.0, 2.0, 3.0]")
    formula = op.to_formula()
    assert formula == "outlier_dist_all"

    # Test from_formula with no class value
    new_op = OutlierDistance.from_formula("outlier_dist")
    assert new_op is not None
    assert new_op.class_value is None

    new_op = OutlierDistance.from_formula(formula)
    assert new_op is not None
    assert new_op.class_value is None

    # Test with specific class value
    op = OutlierDistance(class_value="positive")
    formula = op.to_formula()
    assert formula == "outlier_dist_positive"

    # Test from_formula with specific class value
    new_op = OutlierDistance.from_formula(formula)
    assert new_op is not None
    assert new_op.class_value == "positive"

    # Test invalid formula
    assert OutlierDistance.from_formula("invalid_formula") is None
