import logging
import os

import pandas as pd

from upgini.metadata import ModelTaskType
from upgini.utils.psi import calculate_features_psi, calculate_sparsity_psi

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/psi")


def assert_dicts_equal(actual: dict, expected: dict, precision: int = 4):
    for feature, actual_value in actual.items():
        expected_value = expected[feature]
        assert round(actual_value, precision) == round(
            expected_value, precision
        ), f"Feature {feature}: expected {expected_value}, got {actual_value}"


def test_psi_default_agg():
    df = pd.read_parquet(f"{base_dir}/df_for_psi.parquet")

    cat_features = [
        "f_upgini_ads1_feature13_c64f7cef",
        "f_autofe_date_diff_type1_per_method2_18b92d2f5a",
        "f_autofe_date_diff_min_per_method2_10f7d6c6bf",
        "f_autofe_date_diff_min_per_method2_0152041611",
        "f_autofe_date_diff_type1_abs_bin_25866f7781",
        "f_location_country_postal_nearest_big_city_name_5e49d3ef",
        "f_autofe_date_diff_type1_abs_bin_60ae475431",
        "email_domain",
        "f_autofe_date_diff_type1_abs_bin_01bbab3d7c",
        "f_autofe_date_diff_type1_abs_bin_5b9919ef1f",
        "f_autofe_date_diff_type1_abs_bin_1fa60ce235",
        "f_autofe_date_diff_type1_abs_bin_0ecb6392e7",
    ]

    psi_sparse_values = calculate_sparsity_psi(
        df.drop(columns=["target"]),
        cat_features=cat_features,
        date_column="request_date",
        logger=logging.getLogger(),
        model_task_type=ModelTaskType.BINARY,
    )

    expected_psi_sparse_values = {
        "f_events_date_week_cos1_f6a8c1fc": 0.0,
        "f_economic_date_cpi_pca_6_10ce8957": 0.0,
        "f_upgini_ads4_feature1_02325d70": 0.0016437770344018966,
        "f_upgini_ads8_feature1_39f89fc9": 0.002632707127166154,
        "f_economic_date_cpi_pca_9_3c7905ac": 0.0,
        "f_autofe_date_diff_hist_d_1800_plusinf_673ea0aee6": 0.0,
        "f_upgini_ads116_feature3_19f3bd5d": 0.023077217232259747,
        "f_autofe_date_diff_type1_per_method2_18b92d2f5a": 0.00565599522227576,
        "f_autofe_date_diff_min_per_method2_0152041611": 0.004789708435123827,
        "f_autofe_date_diff_min_per_method2_10f7d6c6bf": 0.0023198792865269592,
        "f_holiday_code_3d_before_d9f28dde": 0.014982429794984766,
        "f_economic_date_cci_pca_7_17261951": 0.0,
        "f_location_country_postal_nearest_big_city_name_5e49d3ef": 0.013611725145487343,
        "f_upgini_ads9_feature117_319972b6": 0.003391344327560647,
        "f_upgini_ads1_feature13_c64f7cef": 0.004379707569734895,
        "f_upgini_ads116_feature2_2a3c1544": 0.017389494679400406,
        "f_upgini_ads17_feature3_be002798": 0.0070716127164706065,
        "f_upgini_ads3_feature7_8c5d5512": 0.017259721870203297,
        "f_upgini_ads115_feature8_7a6aa2e3": 0.003765844472231003,
        "f_events_date_week_sin1_847b5db1": 0.0,
        "f_upgini_ads119_feature3_autofe_emb_outlier_dist_all_584172a4a2": 0.002717490695213861,
        "f_events_date_year_sin7_16d9d7c9": 0.0,
        "f_upgini_ads116_feature4_2d37fd51": 0.026802380447372976,
        "f_upgini_ads9_feature173_e042852f": 0.003391344327560647,
        "f_upgini_ads9_feature196_4e306f7a": 0.003391344327560647,
        "f_upgini_range_ads2_feature2_948a6ce5": 0.004167191081223823,
        "f_upgini_ads60_feature48_fcac0165": 0.002151465859058637,
        "f_upgini_ads12_feature7_bf87912e": 0.011071842844578144,
        "f_upgini_ads17_feature1_9506bef8": 0.0070716127164706065,
        "f_financial_date_crude_oil_gap_b97870de": 0.0,
        "f_events_date_month_cos1_175991a9": 0.0,
        "f_upgini_ads10_feature3_f5b5f17b": 0.011918097120777436,
        "f_upgini_ads6_feature2_5dcbb7cf": 0.0002160546368560826,
        "f_upgini_ads9_feature122_0e053d1d": 0.003391344327560647,
        "f_upgini_ads6_feature5_c3e35ca1": 4.215348552214149e-05,
        "f_upgini_ads9_feature5_7fee0147": 0.003391344327560647,
        "f_upgini_ads5_feature26_2e58ec0d": 0.004214457537337534,
        "f_events_date_year_cos7_9580852c": 0.0,
        "f_events_date_year_sin8_6be64013": 0.0,
        "f_events_date_year_cos6_7c1806a3": 0.0,
        "f_events_date_year_sin9_a4f6646e": 0.0,
        "f_events_date_month_sin3_028c028f": 0.0,
        "f_events_date_year_cos9_f85b1052": 0.0,
        "f_upgini_ads40_feature1_d912f305": 0.0014680358510413766,
        "f_events_date_year_sin10_5261c77c": 0.0,
        "f_financial_date_usd_eur_gap_c8eb8d4a": 0.0,
    }

    assert_dicts_equal(psi_sparse_values, expected_psi_sparse_values)

    psi_values = calculate_features_psi(
        df.drop(columns=["target"]),
        cat_features=cat_features,
        date_column="request_date",
        logger=logging.getLogger(),
        model_task_type=ModelTaskType.BINARY,
    )

    expected_psi_values = {
        "f_events_date_week_cos1_f6a8c1fc": 0.06313013802435638,
        "f_economic_date_cpi_pca_6_10ce8957": 0.0,
        "f_upgini_ads4_feature1_02325d70": 0.013682662343979636,
        "f_upgini_ads8_feature1_39f89fc9": 0.05801911738719491,
        "f_economic_date_cpi_pca_9_3c7905ac": 0.0,
        "f_autofe_catboost_ru_score8_4c3d047cb2": 0.037998337717718135,
        "f_autofe_date_diff_hist_d_1800_plusinf_673ea0aee6": 0.09199854874106064,
        "f_upgini_ads116_feature3_19f3bd5d": 0.049221141550044034,
        "f_autofe_date_diff_type1_per_method2_18b92d2f5a": 0.05694487699301948,
        "f_autofe_date_diff_min_per_method2_0152041611": 0.07275356133015727,
        "f_autofe_date_diff_min_per_method2_10f7d6c6bf": 0.06569605139797896,
        "f_holiday_code_3d_before_d9f28dde": 0.11628451275755555,
        "f_economic_date_cci_pca_7_17261951": 0.0,
        "f_location_country_postal_nearest_big_city_name_5e49d3ef": 0.10589480527779087,
        "f_autofe_date_diff_type1_abs_bin_25866f7781": 0.04368597336214596,
        "f_upgini_ads9_feature117_319972b6": 0.28691235570464463,
        "f_upgini_ads1_feature13_c64f7cef": 0.13845084699614416,
        "f_upgini_ads116_feature2_2a3c1544": 0.06741844241367138,
        "f_upgini_ads17_feature3_be002798": 0.04626903535060746,
        "f_autofe_date_diff_type1_abs_bin_1fa60ce235": 0.018948136147224275,
        "f_upgini_ads3_feature7_8c5d5512": 0.06572526202606854,
        "f_upgini_ads115_feature8_7a6aa2e3": 0.18838711778560838,
        "f_events_date_week_sin1_847b5db1": 0.06313013802435638,
        "f_upgini_ads119_feature3_autofe_emb_outlier_dist_all_584172a4a2": 0.0696422406370498,
        "f_events_date_year_sin7_16d9d7c9": 13.821875074250645,
        "f_upgini_ads116_feature4_2d37fd51": 0.06357565659748789,
        "f_autofe_date_diff_type1_abs_bin_01bbab3d7c": 0.01740336907855491,
        "f_upgini_ads9_feature173_e042852f": 0.2034243222838676,
        "f_upgini_ads9_feature196_4e306f7a": 0.21645767277080172,
        "f_autofe_date_diff_type1_abs_bin_60ae475431": 0.019461763319833465,
        "f_upgini_range_ads2_feature2_948a6ce5": 0.1458550778968836,
        "f_upgini_ads60_feature48_fcac0165": 0.029969897008619503,
        "f_upgini_ads12_feature7_bf87912e": 0.04823386180945783,
        "f_upgini_ads17_feature1_9506bef8": 0.03401604930601499,
        "f_financial_date_crude_oil_gap_b97870de": 0.0,
        "f_events_date_month_cos1_175991a9": 13.968013505004265,
        "f_upgini_ads10_feature3_f5b5f17b": 0.033024442544897205,
        "f_upgini_ads6_feature2_5dcbb7cf": 0.14755979820919685,
        "f_upgini_ads9_feature122_0e053d1d": 0.21386319960680145,
        "f_upgini_ads6_feature5_c3e35ca1": 0.09420048518103294,
        "f_upgini_ads9_feature5_7fee0147": 0.2218842565899956,
        "f_upgini_ads5_feature26_2e58ec0d": 0.04840680506901437,
        "f_autofe_date_diff_type1_abs_bin_5b9919ef1f": 0.03663084701434601,
        "f_events_date_year_cos7_9580852c": 11.558022423547657,
        "f_events_date_year_sin8_6be64013": 5.839559576078206,
        "f_events_date_year_cos6_7c1806a3": 10.450403623379035,
        "f_events_date_year_sin9_a4f6646e": 4.379072102752617,
        "f_events_date_month_sin3_028c028f": 5.252982101493149,
        "f_events_date_year_cos9_f85b1052": 11.093694169437727,
        "f_upgini_ads40_feature1_d912f305": 0.13366719487880735,
        "f_events_date_year_sin10_5261c77c": 12.16231797395598,
        "f_financial_date_usd_eur_gap_c8eb8d4a": 0.0,
        "f_autofe_date_diff_type1_abs_bin_0ecb6392e7": 0.02570474243965017,
    }

    assert_dicts_equal(psi_values, expected_psi_values)


def test_psi_min_agg():
    df = pd.read_parquet(f"{base_dir}/df_for_psi.parquet")

    cat_features = [
        "f_upgini_ads1_feature13_c64f7cef",
        "f_autofe_date_diff_type1_per_method2_18b92d2f5a",
        "f_autofe_date_diff_min_per_method2_10f7d6c6bf",
        "f_autofe_date_diff_min_per_method2_0152041611",
        "f_autofe_date_diff_type1_abs_bin_25866f7781",
        "f_location_country_postal_nearest_big_city_name_5e49d3ef",
        "f_autofe_date_diff_type1_abs_bin_60ae475431",
        "email_domain",
        "f_autofe_date_diff_type1_abs_bin_01bbab3d7c",
        "f_autofe_date_diff_type1_abs_bin_5b9919ef1f",
        "f_autofe_date_diff_type1_abs_bin_1fa60ce235",
        "f_autofe_date_diff_type1_abs_bin_0ecb6392e7",
    ]

    psi_sparse_values = calculate_sparsity_psi(
        df.drop(columns=["target"]),
        cat_features=cat_features,
        date_column="request_date",
        logger=logging.getLogger(),
        model_task_type=ModelTaskType.BINARY,
        stability_agg_func="min",
    )

    expected_psi_sparse_values = {
        "f_events_date_week_cos1_f6a8c1fc": 0.0,
        "f_economic_date_cpi_pca_6_10ce8957": 0.0,
        "f_upgini_ads4_feature1_02325d70": 1.4737529446856481e-06,
        "f_upgini_ads8_feature1_39f89fc9": 8.91676258172046e-05,
        "f_economic_date_cpi_pca_9_3c7905ac": 0.0,
        "f_autofe_date_diff_hist_d_1800_plusinf_673ea0aee6": 0.0,
        "f_upgini_ads116_feature3_19f3bd5d": 0.000870938159275342,
        "f_autofe_date_diff_type1_per_method2_18b92d2f5a": 0.0004763584468303241,
        "f_autofe_date_diff_min_per_method2_0152041611": 0.00039478176382863985,
        "f_autofe_date_diff_min_per_method2_10f7d6c6bf": 0.00014867388922993436,
        "f_holiday_code_3d_before_d9f28dde": 0.0006967783701329189,
        "f_economic_date_cci_pca_7_17261951": 0.0,
        "f_location_country_postal_nearest_big_city_name_5e49d3ef": 0.0005361124871531026,
        "f_upgini_ads9_feature117_319972b6": 3.2539300007843543e-09,
        "f_upgini_ads1_feature13_c64f7cef": 0.0007937226166501034,
        "f_upgini_ads116_feature2_2a3c1544": 0.0019357051536025725,
        "f_upgini_ads17_feature3_be002798": 0.0019688517378633774,
        "f_upgini_ads3_feature7_8c5d5512": 0.002080372912410657,
        "f_upgini_ads115_feature8_7a6aa2e3": 8.026259531062846e-09,
        "f_events_date_week_sin1_847b5db1": 0.0,
        "f_upgini_ads119_feature3_autofe_emb_outlier_dist_all_584172a4a2": 3.431478692018874e-08,
        "f_events_date_year_sin7_16d9d7c9": 0.0,
        "f_upgini_ads116_feature4_2d37fd51": 0.0013189141820604733,
        "f_upgini_ads9_feature173_e042852f": 3.2539300007843543e-09,
        "f_upgini_ads9_feature196_4e306f7a": 3.2539300007843543e-09,
        "f_upgini_range_ads2_feature2_948a6ce5": 1.4318024116435414e-08,
        "f_upgini_ads60_feature48_fcac0165": 7.329761056564374e-05,
        "f_upgini_ads12_feature7_bf87912e": 9.05888003410344e-05,
        "f_upgini_ads17_feature1_9506bef8": 0.0019688517378633774,
        "f_financial_date_crude_oil_gap_b97870de": 0.0,
        "f_events_date_month_cos1_175991a9": 0.0,
        "f_upgini_ads10_feature3_f5b5f17b": 5.083165456954579e-07,
        "f_upgini_ads6_feature2_5dcbb7cf": 4.944875330043538e-06,
        "f_upgini_ads9_feature122_0e053d1d": 3.2539300007843543e-09,
        "f_upgini_ads6_feature5_c3e35ca1": 3.290649690303737e-07,
        "f_upgini_ads9_feature5_7fee0147": 3.2539300007843543e-09,
        "f_upgini_ads5_feature26_2e58ec0d": 0.0003193812451490183,
        "f_events_date_year_cos7_9580852c": 0.0,
        "f_events_date_year_sin8_6be64013": 0.0,
        "f_events_date_year_cos6_7c1806a3": 0.0,
        "f_events_date_year_sin9_a4f6646e": 0.0,
        "f_events_date_month_sin3_028c028f": 0.0,
        "f_events_date_year_cos9_f85b1052": 0.0,
        "f_upgini_ads40_feature1_d912f305": 0.00010089782874292801,
        "f_events_date_year_sin10_5261c77c": 0.0,
        "f_financial_date_usd_eur_gap_c8eb8d4a": 0.0,
    }

    assert_dicts_equal(psi_sparse_values, expected_psi_sparse_values)

    psi_values = calculate_features_psi(
        df.drop(columns=["target"]),
        cat_features=cat_features,
        date_column="request_date",
        logger=logging.getLogger(),
        model_task_type=ModelTaskType.BINARY,
        stability_agg_func="min",
    )

    expected_psi_values = {
        "f_events_date_week_cos1_f6a8c1fc": 0.0016618651798195833,
        "f_economic_date_cpi_pca_6_10ce8957": 0.0,
        "f_upgini_ads4_feature1_02325d70": 1.6468191998645988e-05,
        "f_upgini_ads8_feature1_39f89fc9": 0.00949937259483851,
        "f_economic_date_cpi_pca_9_3c7905ac": 0.0,
        "f_autofe_catboost_ru_score8_4c3d047cb2": 0.006509405063996107,
        "f_autofe_date_diff_hist_d_1800_plusinf_673ea0aee6": 0.012848727854569784,
        "f_upgini_ads116_feature3_19f3bd5d": 0.0031382266131109813,
        "f_autofe_date_diff_type1_per_method2_18b92d2f5a": 0.014558095610139151,
        "f_autofe_date_diff_min_per_method2_0152041611": 0.016861274176790272,
        "f_autofe_date_diff_min_per_method2_10f7d6c6bf": 0.014039697922866313,
        "f_holiday_code_3d_before_d9f28dde": 0.0019041355107546026,
        "f_economic_date_cci_pca_7_17261951": 0.0,
        "f_location_country_postal_nearest_big_city_name_5e49d3ef": 0.027066788250035934,
        "f_autofe_date_diff_type1_abs_bin_25866f7781": 0.0008000980565701312,
        "f_upgini_ads9_feature117_319972b6": 0.009050832745844044,
        "f_upgini_ads1_feature13_c64f7cef": 0.026765923138651643,
        "f_upgini_ads116_feature2_2a3c1544": 0.009216363554149096,
        "f_upgini_ads17_feature3_be002798": 0.014954304305650156,
        "f_autofe_date_diff_type1_abs_bin_1fa60ce235": 0.0010730371561375259,
        "f_upgini_ads3_feature7_8c5d5512": 0.011774790102821632,
        "f_upgini_ads115_feature8_7a6aa2e3": 0.012888378871618061,
        "f_events_date_week_sin1_847b5db1": 0.0016618651798195833,
        "f_upgini_ads119_feature3_autofe_emb_outlier_dist_all_584172a4a2": 0.011004439353660028,
        "f_events_date_year_sin7_16d9d7c9": 0.5553450993047065,
        "f_upgini_ads116_feature4_2d37fd51": 0.004396066443606026,
        "f_autofe_date_diff_type1_abs_bin_01bbab3d7c": 0.0034563752723202204,
        "f_upgini_ads9_feature173_e042852f": 0.00021011684912249242,
        "f_upgini_ads9_feature196_4e306f7a": 0.008860375356981377,
        "f_autofe_date_diff_type1_abs_bin_60ae475431": 0.0026464037161577593,
        "f_upgini_range_ads2_feature2_948a6ce5": 0.0010430102378222107,
        "f_upgini_ads60_feature48_fcac0165": 0.008859040096880821,
        "f_upgini_ads12_feature7_bf87912e": 0.004741319200072894,
        "f_upgini_ads17_feature1_9506bef8": 0.008385209970525156,
        "f_financial_date_crude_oil_gap_b97870de": 0.0,
        "f_events_date_month_cos1_175991a9": 4.344200791006702,
        "f_upgini_ads10_feature3_f5b5f17b": 8.75683267502826e-05,
        "f_upgini_ads6_feature2_5dcbb7cf": 0.00657789404488607,
        "f_upgini_ads9_feature122_0e053d1d": 0.007265057311681455,
        "f_upgini_ads6_feature5_c3e35ca1": 0.025499872020874725,
        "f_upgini_ads9_feature5_7fee0147": 0.00868026187420191,
        "f_upgini_ads5_feature26_2e58ec0d": 0.016228915683739114,
        "f_autofe_date_diff_type1_abs_bin_5b9919ef1f": 0.0029034335911224288,
        "f_events_date_year_cos7_9580852c": 0.18581076016547732,
        "f_events_date_year_sin8_6be64013": 0.13261879116587857,
        "f_events_date_year_cos6_7c1806a3": 0.32004039363152503,
        "f_events_date_year_sin9_a4f6646e": 0.05842551359200361,
        "f_events_date_month_sin3_028c028f": 0.16018803704986403,
        "f_events_date_year_cos9_f85b1052": 0.31951396501718554,
        "f_upgini_ads40_feature1_d912f305": 0.009242249949887106,
        "f_events_date_year_sin10_5261c77c": 0.7187478787256831,
        "f_financial_date_usd_eur_gap_c8eb8d4a": 0.0,
        "f_autofe_date_diff_type1_abs_bin_0ecb6392e7": 0.002722524598683429,
    }

    assert_dicts_equal(psi_values, expected_psi_values)
