import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.binary import (
    JaroWinklerSim1,
    JaroWinklerSim2,
    LevenshteinSim,
)
from upgini.autofe.feature import Feature


def test_parse_sim():
    meta = {
        "formula": "sim_jw2(msisdn_ads_1617304852_education_phone_competition_role,telegram_ads_full_1841355682_web_phone_top_country_fb)",  # noqa: E501
        "display_index": "38aa5abe",
        "base_columns": [
            {
                "original_name": "msisdn_ads_1617304852_education_phone_competition_role",
                "hashed_name": "f_education_phone_competition_role_bfa11e4f",
                "ads_definition_id": "53c369e5-a486-4ebf-877c-74282897283f",
                "is_augmented": "false",
            },
            {
                "original_name": "telegram_ads_full_1841355682_web_phone_top_country_fb",
                "hashed_name": "f_web_phone_top_country_fb_05ae58ec",
                "ads_definition_id": "0edcc714-555a-4734-9ab2-d34bce71300e",
                "is_augmented": "false",
            },
        ],
        "operator_params": {"lang": "en", "model_type": "fasttext"},
    }
    feature = Feature.from_formula(meta["formula"]).set_display_index(meta["display_index"])
    assert feature.get_op_display_name() == "sim_jw2"

    df = pd.DataFrame(
        {
            "msisdn_ads_1617304852_education_phone_competition_role": ["test1", "test2", "test3"],
            "telegram_ads_full_1841355682_web_phone_top_country_fb": ["test1", "test23", "test345"],
        }
    )
    result = feature.calculate(df)

    expected_result = pd.Series([1.0, 0.944444, 0.904762])
    print(result)
    assert_series_equal(expected_result, result, atol=10**-6)


def test_string_sim():
    data = pd.DataFrame(
        [
            ["book", "look"],
            ["blow", None],
            [None, "Jeremy"],
            ["below", "bewoll"],
            [None, None],
            ["abc", "abc"],
            ["four", "seven"],
        ],
        columns=["a", "b"],
    )

    expected_jw1 = pd.Series([0.833, None, None, 0.902, None, 1.0, 0.0])
    expected_jw2 = pd.Series([0.883, None, None, 0.739, None, 1.0, 0.0])
    expected_lv = pd.Series([0.75, None, None, 0.5, None, 1.0, 0.0])

    assert_series_equal(JaroWinklerSim1().calculate_binary(data["a"], data["b"]).round(3), expected_jw1)
    assert_series_equal(JaroWinklerSim2().calculate_binary(data["a"], data["b"]).round(3), expected_jw2)
    assert_series_equal(LevenshteinSim().calculate_binary(data["a"], data["b"]).round(3), expected_lv)
