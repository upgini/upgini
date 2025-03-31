from typing import Dict

import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.feature import Feature
from upgini.autofe.unary import Norm


def test_norm():
    data = pd.DataFrame(
        [
            [None, 1, None],
            [1, 222, None],
            [333, 4, None],
            [1, 2, None],
            [3, 4, None],
            [0, 1, None],
            [1, 0, None],
            [2, 3, None],
            [3, 2, None],
            [1, None, None],
        ],
        columns=["a", "b", "c"],
        index=[9, 8, 7, 6, 5, 1, 2, 3, 4, 0],
    )

    expected_result = pd.DataFrame(
        [
            [None, 0.00450218, None],
            [0.00300266, 0.99948299, None],
            [0.99988729, 0.0180087, None],
            [0.00300266, 0.00900435, None],
            [0.00900799, 0.0180087, None],
            [0.0, 0.00450218, None],
            [0.00300266, 0.0, None],
            [0.00600533, 0.01350653, None],
            [0.00900799, 0.00900435, None],
            [0.00300266, None, None],
        ],
        columns=["a", "b", "c"],
        index=[9, 8, 7, 6, 5, 1, 2, 3, 4, 0],
    )

    operand_by_column: Dict[str, Norm] = {}
    for c in data.columns:
        operand = Norm()
        operand_by_column[c] = operand
        assert_series_equal(operand.calculate_unary(data[c]), expected_result[c])

    transform_data = pd.DataFrame(
        {
            "a": [None, 1, 50, 100],
            "b": [1, 50, 100, None],
            "c": [1, 2, 3, 4],
        }
    )

    expected_transform_result = pd.DataFrame(
        {
            "a": [None, 0.00300266, 0.15013255, 0.30026510],
            "b": [0.00450218, 0.22510878, 0.45021756, None],
            "c": [0.18257418, 0.36514837, 0.54772255, 0.73029674],
        }
    )

    for c in transform_data.columns:
        operand = operand_by_column[c]
        assert_series_equal(operand.calculate_unary(transform_data[c]), expected_transform_result[c])


def test_deserialization_norm():
    formulae = {
        "formula": "(norm(postal_code_landuse_2km_area_413200106_location_country_postal_residential_2km_area_to_postal_area)*postal_code_poi_5km_cnt_1046005976_location_country_postal_poi_accommodation_guest_house_5km_cnt)",  # noqa: E501
        "display_index": "9d938aac",
        "base_columns": [
            {
                "original_name": "postal_code_landuse_2km_area_413200106_location_country_postal_residential_2km_area_to_postal_area",  # noqa: E501
                "hashed_name": "f_location_country_postal_residential_2km_area_to_postal_area_c695d4eb",
                "ads_definition_id": "4cc902f7-7768-469e-a5e1-0f9f18d16dd5",
                "is_augmented": False,
            },
            {
                "original_name": "postal_code_poi_5km_cnt_1046005976_location_country_postal_poi_accommodation_guest_house_5km_cnt",  # noqa: E501
                "hashed_name": "f_location_country_postal_poi_accommodation_guest_house_5km_cnt_8d1d59d8",
                "ads_definition_id": "e4d418e0-e6b0-4d2d-acd2-09186efef827",
                "is_augmented": False,
            },
        ],
        "operator_params": {"norm": "1741.9150381117904", "alias": "mul"},
    }
    feature = (
        Feature.from_formula(formulae["formula"])
        .set_display_index(formulae["display_index"])
        .set_op_params(formulae["operator_params"])
    )
    df = pd.DataFrame(
        {
            "postal_code_landuse_2km_area_413200106_location_country_postal_residential_2km_area_to_postal_area": [
                0.58532
            ],
            "postal_code_poi_5km_cnt_1046005976_location_country_postal_poi_accommodation_guest_house_5km_cnt": [2],
        }
    )
    result = feature.calculate(df)
    print(result)
    assert result.values[0] == 0.0006720419620861393
