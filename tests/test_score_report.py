import base64
import json
from pathlib import Path

import pandas as pd
from requests_mock.mocker import Mocker

from upgini.features_enricher import FeaturesEnricher
from upgini.metadata import (
    BaseColumnMetadata,
    FeaturesMetadataV2,
    GeneratedFeatureMetadata,
    ModelTaskType,
    ProviderTaskMetadataV2,
    SearchKey,
)
from upgini.report.assemble import (
    UPGINI_REPORT_BRANDING_URL,
    assemble_report_data,
    build_search_results,
    format_search_duration,
)
from upgini.report.data import FeatureRow
from upgini.report.html import generate_html_report
from upgini.resource_bundle import bundle
from upgini.search_task import SearchTask
from upgini.utils.feature_info import CLIENT_SOURCE, GENERATED_SOURCE

from .utils import mock_default_requests

SearchTask.PROTECT_FROM_RATE_LIMIT = False


def _ensemble_metadata(ensemble_col: str) -> ProviderTaskMetadataV2:
    return ProviderTaskMetadataV2(
        features=[
            FeaturesMetadataV2(
                name=ensemble_col,
                type="numeric",
                source="generated",
                hit_rate=100.0,
                shap_value=1.0,
                commercial_schema="Trial",
                data_provider="Upgini",
                data_source="AutoFE: features from Usage Data",
            ),
            FeaturesMetadataV2(
                name="f_model1_abc",
                type="numeric",
                source="ads",
                hit_rate=98.7,
                shap_value=0.08,
                data_provider="Upgini",
                data_source="Usage Data",
                psi_value=0.05,
                drift_score=0.03,
            ),
            FeaturesMetadataV2(
                name="f_model2_xyz",
                type="numeric",
                source="ads",
                hit_rate=96.9,
                shap_value=-0.15,
                data_provider="Upgini",
                data_source="Accounts Availability",
                psi_value=0.22,
                drift_score=0.11,
            ),
            FeaturesMetadataV2(
                name="f_autofe_div",
                type="numeric",
                source="generated",
                hit_rate=97.8,
                shap_value=0.066,
                commercial_schema="Trial",
                data_provider="Upgini",
                data_source=(
                    "AutoFE: features from <a href='https://upgini.com/#data_sources' "
                    "target='_blank' rel='noopener noreferrer'>POI data OpenStreetMap</a>"
                ),
                psi_value=0.0,
            ),
            FeaturesMetadataV2(
                name="children_num_7967cb",
                type="numeric",
                source="etalon",
                hit_rate=100.0,
                shap_value=0.0,
            ),
            FeaturesMetadataV2(
                name="pd002_6e6a41",
                type="numeric",
                source="etalon",
                hit_rate=100.0,
                shap_value=0.73,
                psi_value=0.0,
            ),
            FeaturesMetadataV2(
                name="datetime_day_in_quarter_sin_65d4f7",
                type="numeric",
                source="generated",
                hit_rate=100.0,
                shap_value=0.0,
                data_provider="Upgini",
                data_source="LLM with external data augmentation",
            ),
        ],
        joined_ads_features_count=1234,
        joined_ads_count=17,
        generated_features=[
            GeneratedFeatureMetadata(
                alias="upgini_score",
                formula="ensemble_score(model1,model2)",
                display_index="abc123",
                base_columns=[
                    BaseColumnMetadata(
                        original_name="children_num", hashed_name="children_num_7967cb", is_augmented=False
                    ),
                    BaseColumnMetadata(
                        original_name="datetime_day_in_quarter_sin",
                        hashed_name="datetime_day_in_quarter_sin_65d4f7",
                        is_augmented=False,
                    ),
                ],
            ),
            GeneratedFeatureMetadata(
                formula="catboost_score(f_model1_abc,pd002_6e6a41,f_autofe_div)",
                display_index="fold1",
                base_columns=[
                    BaseColumnMetadata(
                        original_name="f_model1_abc",
                        hashed_name="f_model1_abc",
                        ads_definition_id="ads-usage",
                        is_augmented=False,
                    ),
                    BaseColumnMetadata(
                        original_name="pd002_6e6a41", hashed_name="pd002_6e6a41", is_augmented=False
                    ),
                    BaseColumnMetadata(
                        original_name="f_autofe_div", hashed_name="f_autofe_div", is_augmented=False
                    ),
                ],
            ),
            GeneratedFeatureMetadata(
                formula="catboost_score(f_model2_xyz,pd002_6e6a41,f_autofe_div)",
                display_index="fold2",
                base_columns=[
                    BaseColumnMetadata(
                        original_name="f_model2_xyz",
                        hashed_name="f_model2_xyz",
                        ads_definition_id="ads-accounts",
                        is_augmented=False,
                    ),
                    BaseColumnMetadata(
                        original_name="pd002_6e6a41", hashed_name="pd002_6e6a41", is_augmented=False
                    ),
                    BaseColumnMetadata(
                        original_name="f_autofe_div", hashed_name="f_autofe_div", is_augmented=False
                    ),
                ],
            ),
            GeneratedFeatureMetadata(
                alias="div",
                formula="(a/b)",
                display_index="",
                base_columns=[
                    BaseColumnMetadata(
                        original_name="a",
                        hashed_name="f_location_a",
                        ads_definition_id="ads-osm",
                        is_augmented=True,
                    ),
                    BaseColumnMetadata(
                        original_name="b",
                        hashed_name="f_location_b",
                        ads_definition_id="ads-osm",
                        is_augmented=True,
                    ),
                ],
            ),
        ],
    )


def _parse_report_data(html: str) -> dict:
    payload_start = html.index("const REPORT_DATA = ") + len("const REPORT_DATA = ")
    payload_end = html.index(";\n\n/* =")
    return json.loads(html[payload_start:payload_end])


def test_format_search_duration():
    assert format_search_duration(None) is None
    assert format_search_duration(9) == "9 sec"
    assert format_search_duration(62) == "1 min 2 sec"
    assert format_search_duration(3723) == "1 h 2 min 3 sec"


def test_report_without_branding_url_shows_only_upgini(monkeypatch):
    monkeypatch.delenv(UPGINI_REPORT_BRANDING_URL, raising=False)
    data = assemble_report_data(search_id="search-abc", search_keys=["PHONE"], bundle=bundle)
    assert data.metadata.logo_url is None
    html = generate_html_report(data)
    payload = _parse_report_data(html)
    assert "partnerLogo" not in payload["meta"]
    assert 'aria-label="Upgini"' in html


def test_report_uses_branding_url_from_env(monkeypatch):
    monkeypatch.setenv(UPGINI_REPORT_BRANDING_URL, " https://cdn.example.com/logo.svg ")
    data = assemble_report_data(search_id="search-abc", search_keys=["PHONE"], bundle=bundle)
    assert data.metadata.logo_url == "https://cdn.example.com/logo.svg"
    html = generate_html_report(data)
    payload = _parse_report_data(html)
    assert payload["meta"]["partnerLogo"]["src"] == "https://cdn.example.com/logo.svg"
    assert 'aria-label="Upgini"' in html


def test_generate_html_report_from_assembled_data():
    metrics_df = pd.DataFrame(
        {
            bundle.get("quality_metrics_segment_header"): ["Train", "Eval 1"],
            bundle.get("quality_metrics_rows_header"): [100, 40],
            bundle.get("quality_metrics_mean_target_header"): [0.31, 0.28],
            bundle.get("quality_metrics_baseline_header").format("GINI"): [0.4, 0.39],
            bundle.get("quality_metrics_enriched_header").format("GINI"): [0.512, 0.48],
            bundle.get("quality_metrics_uplift_header"): [0.112, 0.09],
            bundle.get("quality_metrics_uplift_perc_header"): ["28.0%", "23.1%"],
        }
    )
    data = assemble_report_data(
        search_id="search-abc",
        search_keys=["PHONE", "DATE"],
        search_duration_seconds=125,
        reference_rows=140,
        samples=["Train", "Eval 1"],
        metrics_df=metrics_df,
        metric_name="GINI",
        model_features=2,
        is_binary=True,
        bundle=bundle,
    )
    html = generate_html_report(data)
    payload = _parse_report_data(html)

    assert payload["meta"]["searchId"] == "search-abc"
    assert payload["meta"]["searchKeys"] == ["PHONE", "DATE"]
    assert payload["meta"]["searchDuration"] == "2 min 5 sec"
    assert payload["meta"]["totalRows"] == 140
    assert payload["keyResult"]["metrics"]["gini"]["bySample"]["train"]["enriched"] == 0.512
    assert payload["samples"][0] == {
        "id": "train",
        "label": "Train",
        "caption": "OOF validation",
        "color": "#1645ee",
    }
    assert payload["samples"][1]["caption"] == ""
    assert payload["summaryCards"][1] == {"label": "Used in model", "value": "2", "caption": ""}
    assert payload["summaryCards"][2]["caption"] == ""
    assert payload["summaryCards"][3]["caption"] == ""
    assert payload["sampleStats"]["rows"][1]["values"]["train"] == "100"

    assert "const REPORT_DATA =" in html
    assert "__REPORT_DATA__" not in html
    assert "Download HTML" in html
    assert "FIT completed successfully" in html
    assert "Features found" in html
    assert "Count 1's (target)" in html
    assert data.summary.model_features == 2
    assert data.summary.relevant_features is None
    assert "Sample stats" in html
    assert "Performance" in html
    assert "Score analysis" in html
    assert "Score distribution" in html
    assert "Score stability (PSI)" in html
    assert "reference: first period (baseline month)" in html
    assert "Features SHAP" in html
    assert "Search results" in html
    assert "f_autofe_ensemble_score_abc123" not in html
    assert data.quality_by_sample[0].metric == "GINI"
    assert data.quality_by_sample[0].enriched == 0.512
    assert data.quality_by_sample[0].std is None
    assert data.sample_stats[0].rows == 100
    assert data.sample_stats[0].mean_target == 0.31
    assert data.features == []
    assert data.sources == []
    assert data.autofe == []


def test_quality_sample_uses_separate_std_column():
    metrics_df = pd.DataFrame(
        {
            bundle.get("quality_metrics_segment_header"): ["Train"],
            bundle.get("quality_metrics_enriched_header").format("GINI"): [0.512],
            bundle.get("quality_metrics_enriched_std_header").format("GINI"): [0.010],
        }
    )
    data = assemble_report_data(
        search_id="search-abc",
        search_keys=["PHONE"],
        metrics_df=metrics_df,
        metric_name="GINI",
        bundle=bundle,
    )
    assert data.quality_by_sample[0].enriched == 0.512
    assert data.quality_by_sample[0].std == 0.010
    html = generate_html_report(data)
    assert '"enriched": 0.512' in html
    assert '"enrichedCi": 0.01' in html


def test_jupyter_metrics_table_combines_std():
    from upgini.metrics import format_display_metric, format_metrics_for_display

    assert format_display_metric(0.512, 0.010) == "0.512 ± 0.010"
    raw = pd.DataFrame(
        {
            bundle.get("quality_metrics_enriched_header").format("GINI"): [0.512],
            bundle.get("quality_metrics_enriched_std_header").format("GINI"): [0.010],
        }
    )
    display = format_metrics_for_display(raw, "GINI", bundle)
    assert display.loc[0, "Enriched GINI"] == "0.512 ± 0.010"
    assert "Enriched GINI std" not in display.columns


def _ensemble_enricher(url: str, ensemble_col: str = "f_autofe_upgini_score_abc123") -> FeaturesEnricher:
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        logs_enabled=False,
    )
    enricher._search_task = SearchTask("search-abc")
    enricher._search_task.provider_metadata_v2 = [_ensemble_metadata(ensemble_col)]
    enricher.feature_names_ = [ensemble_col]
    enricher.search_duration_seconds = 42
    enricher.metrics = pd.DataFrame(
        {
            bundle.get("quality_metrics_segment_header"): ["Train"],
            bundle.get("quality_metrics_enriched_header").format("GINI"): ["0.610"],
        }
    )
    enricher.metrics_metric_name = "GINI"
    return enricher


def test_score_report_html_is_not_written_to_disk(requests_mock: Mocker, tmp_path: Path, monkeypatch):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    monkeypatch.chdir(tmp_path)
    enricher = _ensemble_enricher(url)

    html = enricher._score_report_html()
    payload = _parse_report_data(html)

    assert html is not None
    assert "search-abc" in html
    assert '"PHONE"' in html
    assert "SearchKey" not in html
    assert '"enriched": 0.61' in html
    assert "42 sec" in html
    assert payload["summaryCards"][1] == {
        "label": "Used in model",
        "value": "4",
        "caption": "2 external · 1 original · 1 AutoFE",
    }
    assert payload["summaryCards"][0]["value"] == "1234"
    assert not (tmp_path / "reports").exists()
    assert list(tmp_path.glob("*.html")) == []


def test_html_report_button_does_not_build_pdf(requests_mock: Mocker, tmp_path: Path, monkeypatch):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    monkeypatch.chdir(tmp_path)
    enricher = _ensemble_enricher(url)
    pdf_calls = []
    opened = []
    monkeypatch.setattr("upgini.features_enricher.ipython_available", lambda: True)
    monkeypatch.setattr(
        "upgini.features_enricher.show_button_open_report",
        lambda source, **kwargs: opened.append((source, kwargs.get("download_name"))) or "html",
    )
    monkeypatch.setattr(
        "upgini.features_enricher.prepare_and_show_report",
        lambda *args, **kwargs: pdf_calls.append(True) or "pdf",
    )

    assert enricher._FeaturesEnricher__show_report_button() == "html"
    assert pdf_calls == []
    assert opened[0][1] == "upgini-report-search-abc.html"
    assert "search-abc" in opened[0][0]
    assert not (tmp_path / "reports").exists()
    assert list(tmp_path.glob("*.html")) == []
    assert list(tmp_path.glob("*.pdf")) == []


def test_ordinary_report_button_downloads_pdf(requests_mock: Mocker, monkeypatch):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    enricher = FeaturesEnricher(endpoint=url, logs_enabled=False)
    enricher._search_task = SearchTask("search-abc")
    enricher.feature_names_ = ["ads_feature"]
    enricher.report_html = "<html>stale ensemble report</html>"
    pdf_calls = []
    opened = []
    monkeypatch.setattr("upgini.features_enricher.ipython_available", lambda: True)
    monkeypatch.setattr(
        "upgini.features_enricher.show_button_open_report",
        lambda source, **kwargs: opened.append(source) or "html",
    )
    monkeypatch.setattr(
        "upgini.features_enricher.prepare_and_show_report",
        lambda *args, **kwargs: pdf_calls.append(True) or "pdf",
    )

    assert enricher._has_single_ensemble_score() is False
    assert enricher._FeaturesEnricher__show_report_button() == "pdf"
    assert pdf_calls == [True]
    assert opened == []


def test_single_ensemble_score_allows_etalon_features(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    ensemble_col = "f_autofe_upgini_score_abc123"
    enricher = _ensemble_enricher(url, ensemble_col)
    enricher.X = None
    enricher.fit_columns_renaming = {"client_feature_8ddf40": "client_feature"}
    enricher.feature_names_ = [ensemble_col, "client_feature"]
    meta = _ensemble_metadata(ensemble_col)
    meta.features.append(
        FeaturesMetadataV2(
            name="client_feature_8ddf40",
            type="numeric",
            source="etalon",
            hit_rate=100.0,
            shap_value=0.0,
        )
    )
    enricher._search_task.provider_metadata_v2 = [meta]

    assert enricher._has_single_ensemble_score()
    assert enricher._get_single_ensemble_score_name() == ensemble_col
    assert enricher._score_report_html() is not None


def test_skip_oot_psi_with_ensemble_and_client_features(requests_mock: Mocker, monkeypatch):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    ensemble_col = "f_autofe_upgini_score_abc123"
    enricher = _ensemble_enricher(url, ensemble_col)
    enricher.feature_names_ = [ensemble_col, "client_feature"]
    enricher.external_source_feature_names = []
    checked = []
    monkeypatch.setattr(enricher, "_check_stability", lambda *args, **kwargs: checked.append(True) or set())

    enricher._select_features_by_psi(
        X=pd.DataFrame({"client_feature": [1, 2], "phone": [3, 4]}),
        y=pd.Series([0, 1]),
        eval_set=None,
        stability_threshold=0.2,
        stability_agg_func="max",
    )

    assert enricher._has_single_ensemble_score()
    assert checked == []
    assert enricher._is_extra_enriched_feature("f_model1_abc")
    assert enricher._is_extra_enriched_feature("f_autofe_div")
    assert enricher._is_extra_enriched_feature("pd002_6e6a41") is False


def test_oot_psi_not_skipped_with_extra_ads_features(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    ensemble_col = "f_autofe_upgini_score_abc123"
    enricher = _ensemble_enricher(url, ensemble_col)
    enricher.feature_names_ = [ensemble_col, "f_model1_abc"]
    enricher.external_source_feature_names = ["f_model1_abc"]

    assert enricher._has_single_ensemble_score() is False


def test_single_ensemble_score_rejects_extra_ads_features(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    ensemble_col = "f_autofe_upgini_score_abc123"
    enricher = _ensemble_enricher(url, ensemble_col)
    enricher.feature_names_ = [ensemble_col, "f_model1_abc"]

    assert enricher._has_single_ensemble_score() is False
    assert enricher._score_report_html() is None


def test_score_report_skipped_without_ensemble(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    enricher = FeaturesEnricher(endpoint=url, logs_enabled=False)
    enricher._search_task = SearchTask("search-abc")
    enricher.feature_names_ = ["ads_feature"]

    assert enricher._score_report_html() is None


def test_get_ensemble_score_column_does_not_need_baseline(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    ensemble_col = "f_autofe_upgini_score_abc123"
    fitting_X = pd.DataFrame({"client_feature": [0.1, 0.8]})
    fitting_enriched_X = fitting_X.copy()
    fitting_enriched_X[ensemble_col] = [0.2, 0.9]

    enricher = FeaturesEnricher(endpoint=url, logs_enabled=False)
    enricher._search_task = SearchTask("fake_search")
    enricher._search_task.provider_metadata_v2 = [_ensemble_metadata(ensemble_col)]

    assert enricher.baseline_score_column is None
    assert enricher._get_ensemble_score_column(fitting_X, fitting_enriched_X) == ensemble_col


def test_assemble_report_data_uses_dataset_and_cached_scores(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    ensemble_col = "f_autofe_upgini_score_abc123"
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE, "date": SearchKey.DATE},
        endpoint=url,
        logs_enabled=False,
    )
    enricher._search_task = SearchTask("search-abc")
    enricher._search_task.provider_metadata_v2 = [_ensemble_metadata(ensemble_col)]
    enricher.feature_names_ = [ensemble_col]
    enricher.metrics_metric_name = "GINI"
    enricher.model_task_type = ModelTaskType.BINARY
    enricher.fit_search_keys = enricher.search_keys
    enricher.X = pd.DataFrame(
        {
            "phone": [1, 2, 3, 4],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"]),
        }
    )
    enricher.y = pd.Series([0, 1, 0, 1])
    eval_x = pd.DataFrame(
        {
            "phone": [5, 6],
            "date": pd.to_datetime(["2024-02-01", "2024-02-02"]),
        }
    )
    eval_y = pd.Series([0, 1])
    enricher.eval_set = [(eval_x, eval_y)]
    enricher.df_with_original_index = pd.concat(
        [
            enricher.X.assign(target=enricher.y.to_numpy(), eval_set_index=0),
            eval_x.assign(target=eval_y.to_numpy(), eval_set_index=1),
        ],
        ignore_index=True,
    )
    sampled_x = enricher.X.copy()
    enriched_x = sampled_x.copy()
    enriched_x[ensemble_col] = [0.1, 0.9, 0.2, 0.8]
    eval_enriched = eval_x.copy()
    eval_enriched[ensemble_col] = [0.15, 0.85]
    enricher._FeaturesEnricher__cached_sampled_datasets["hash"] = (
        sampled_x,
        enricher.y.copy(),
        enriched_x,
        {0: (eval_x.copy(), eval_enriched, eval_y.copy())},
        enricher.search_keys,
        {},
        [],
    )

    data = enricher._assemble_report_data()

    assert [stats.sample for stats in data.sample_stats] == ["Train", "Eval 1"]
    assert data.sample_stats[0].rows == 4
    assert data.sample_stats[1].rows == 2
    assert data.charts.timeline_months == ["Jan 2024", "Feb 2024"]
    assert data.charts.quality_monthly[0].enriched[0] is not None
    assert data.charts.histograms[0].n_target_0 == 2
    assert data.charts.score_psi[0].rows[0] == 2
    assert [row.name for row in data.features] == [
        "pd002_6e6a41",
        "f_model2_xyz",
        "f_model1_abc",
        "f_autofe_div",
    ]
    assert data.summary.model_features == 4
    assert data.summary.relevant_features == 3
    assert data.summary.features_found == 1234
    assert data.summary.joined_sources == 17
    assert data.summary.external_features == 2
    assert data.summary.original_features == 1
    assert data.summary.autofe_features == 1
    assert data.summary.contributed_sources == 3


def test_ensemble_html_report_lists_model_rows_not_score(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    ensemble_col = "f_autofe_upgini_score_abc123"
    enricher = _ensemble_enricher(url, ensemble_col)

    assert enricher._has_single_ensemble_score()
    assert enricher.feature_names_ == [ensemble_col]

    data = enricher._assemble_report_data()
    html = generate_html_report(data)
    payload = _parse_report_data(html)
    names = [row["name"] for row in payload["features"]]
    by_name = {row["name"]: row for row in payload["features"]}

    assert names == ["pd002_6e6a41", "f_model2_xyz", "f_model1_abc", "f_autofe_div"]
    assert payload["searchResults"]["relevantFeaturesCount"] == "4"
    assert payload["searchResults"]["autofeCount"] == "1"
    assert ensemble_col not in names
    assert "children_num_7967cb" not in names
    assert "datetime_day_in_quarter_sin_65d4f7" not in names
    assert "children_num" not in names
    assert by_name["pd002_6e6a41"]["shap"] == 0.73
    assert by_name["pd002_6e6a41"]["provider"] == CLIENT_SOURCE
    assert by_name["pd002_6e6a41"]["shapClass"] == "user"
    assert by_name["pd002_6e6a41"]["source"] == CLIENT_SOURCE
    assert by_name["f_model2_xyz"]["shap"] == -0.15
    assert by_name["f_model2_xyz"]["importance"] == 0.15
    assert by_name["f_model2_xyz"]["status"] == "watch"
    assert by_name["f_model2_xyz"]["provider"] == "Upgini"
    assert by_name["f_model2_xyz"]["shapClass"] == "upgini"
    assert by_name["f_model2_xyz"]["source"] == "Accounts Availability"
    assert by_name["f_autofe_div"]["shap"] == 0.066
    assert by_name["f_autofe_div"]["provider"] == "Upgini"
    assert by_name["f_autofe_div"]["shapClass"] == "autofe"
    assert by_name["f_autofe_div"]["source"] == (
        "AutoFE: features from <a href='https://upgini.com/#data_sources' "
        "target='_blank' rel='noopener noreferrer'>POI data OpenStreetMap</a>"
    )
    assert payload["summaryCards"][0] == {"label": "Features found", "value": "1234", "caption": ""}
    assert payload["summaryCards"][1] == {
        "label": "Used in model",
        "value": "4",
        "caption": "2 external · 1 original · 1 AutoFE",
    }
    assert payload["summaryCards"][2] == {"label": "Data sources", "value": "17", "caption": "3 contributed"}
    assert payload["summaryCards"][3] == {"label": "Stability", "value": "75%", "caption": "3 of 4 are stable"}
    assert [row["generatedFeature"] for row in payload["searchResults"]["autofe"]] == ["f_autofe_div"]
    assert payload["searchResults"]["autofe"][0]["sources"] == (
        "<a href='https://upgini.com/#data_sources' target='_blank' rel='noopener noreferrer'>POI data OpenStreetMap</a>"
    )
    assert payload["searchResults"]["autofe"][0]["sourceFeatures"] == "f_location_a, f_location_b"
    assert payload["searchResults"]["autofe"][0]["functions"] == "/"
    assert {row["source"] for row in payload["searchResults"]["sources"]} == {
        "Accounts Availability",
        "Usage Data",
        (
            "AutoFE: features from <a href='https://upgini.com/#data_sources' "
            "target='_blank' rel='noopener noreferrer'>POI data OpenStreetMap</a>"
        ),
    }
    assert data.model_feature_shap[0].feature == "pd002_6e6a41"
    assert data.model_feature_shap[0].mean_abs_shap == 0.73
    assert data.model_feature_shap[1].feature == "f_model2_xyz"
    assert "ensemble_score(model1,model2)" not in html
    assert ensemble_col not in html


def test_ensemble_autofe_tab_lists_nested_formula_features(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    ensemble_col = "f_autofe_upgini_score_abc123"
    hashed_autofe = "f_autofe_div_57e6c58173"
    enricher = _ensemble_enricher(url, ensemble_col)
    meta = _ensemble_metadata(ensemble_col)
    for feature in meta.features:
        if feature.name == "f_autofe_div":
            feature.name = hashed_autofe
    meta.generated_features = [
        GeneratedFeatureMetadata(
            alias="upgini_score",
            formula=(
                "ensemble_score("
                "catboost_score(f_model1_abc,pd002_6e6a41,(a/b)),"
                "catboost_score(f_model2_xyz,pd002_6e6a41,(a/b)))"
            ),
            display_index="abc123",
            base_columns=[
                BaseColumnMetadata(
                    original_name="a",
                    hashed_name="f_location_a",
                    ads_definition_id="ads-osm",
                    is_augmented=True,
                ),
                BaseColumnMetadata(
                    original_name="b",
                    hashed_name="f_location_b",
                    ads_definition_id="ads-osm",
                    is_augmented=True,
                ),
                BaseColumnMetadata(
                    original_name="f_model1_abc",
                    hashed_name="f_model1_abc",
                    ads_definition_id="ads-usage",
                    is_augmented=False,
                ),
                BaseColumnMetadata(
                    original_name="pd002_6e6a41", hashed_name="pd002_6e6a41", is_augmented=False
                ),
                BaseColumnMetadata(
                    original_name="f_model2_xyz",
                    hashed_name="f_model2_xyz",
                    ads_definition_id="ads-accounts",
                    is_augmented=False,
                ),
            ],
        )
    ]
    enricher._search_task.provider_metadata_v2 = [meta]

    data = enricher._assemble_report_data()
    payload = _parse_report_data(generate_html_report(data))

    assert [row["generatedFeature"] for row in payload["searchResults"]["autofe"]] == [hashed_autofe]
    assert payload["searchResults"]["autofe"][0]["sourceFeatures"] == "f_location_a, f_location_b"
    assert payload["searchResults"]["autofe"][0]["functions"] == "/"
    assert payload["searchResults"]["autofeCount"] == "1"


def test_ordinary_autofe_descriptions_match_display_suffix(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        logs_enabled=False,
    )
    enricher._search_task = SearchTask("search-abc")
    enricher._search_task.provider_metadata_v2 = [
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(
                    name="f_autofe_div_18b92d2f5a",
                    type="numeric",
                    source="generated",
                    hit_rate=97.8,
                    shap_value=0.12,
                    data_provider="Upgini",
                    data_source="AutoFE: features from Usage Data",
                ),
                FeaturesMetadataV2(
                    name="client_feat",
                    type="numeric",
                    source="etalon",
                    hit_rate=100.0,
                    shap_value=0.4,
                ),
            ],
            generated_features=[
                GeneratedFeatureMetadata(
                    alias="div",
                    formula="(a/b)",
                    display_index="18b92d2f5a",
                    base_columns=[
                        BaseColumnMetadata(
                            original_name="a", hashed_name="f_location_a", is_augmented=True
                        ),
                        BaseColumnMetadata(
                            original_name="b", hashed_name="f_location_b", is_augmented=True
                        ),
                    ],
                )
            ],
        )
    ]

    df = enricher.get_autofe_features_description()
    assert df is not None
    assert list(df[bundle.get("autofe_descriptions_feature_name")]) == ["f_autofe_div_18b92d2f5a"]
    assert list(df[bundle.get("autofe_descriptions_feature").format(1)]) == ["f_location_a"]
    assert list(df[bundle.get("autofe_descriptions_feature").format(2)]) == ["f_location_b"]
    assert list(df[bundle.get("autofe_descriptions_function")]) == ["/"]
    assert enricher._has_single_ensemble_score() is False


def test_ordinary_autofe_skips_vector_ops_and_more_than_two_base_columns(requests_mock: Mocker):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        logs_enabled=False,
    )
    enricher._search_task = SearchTask("search-abc")
    enricher._search_task.provider_metadata_v2 = [
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(
                    name="f_autofe_mean_1",
                    type="numeric",
                    source="generated",
                    hit_rate=97.8,
                    shap_value=0.2,
                    data_source="AutoFE: features from Usage Data",
                ),
                FeaturesMetadataV2(
                    name="f_autofe_div_wide",
                    type="numeric",
                    source="generated",
                    hit_rate=97.8,
                    shap_value=0.3,
                    data_source="AutoFE: features from Usage Data",
                ),
            ],
            generated_features=[
                GeneratedFeatureMetadata(
                    formula="mean(a,b,c)",
                    display_index="1",
                    base_columns=[
                        BaseColumnMetadata(original_name="a", hashed_name="a", is_augmented=True),
                        BaseColumnMetadata(original_name="b", hashed_name="b", is_augmented=True),
                        BaseColumnMetadata(original_name="c", hashed_name="c", is_augmented=True),
                    ],
                ),
                GeneratedFeatureMetadata(
                    alias="div",
                    formula="(a/b)",
                    display_index="wide",
                    base_columns=[
                        BaseColumnMetadata(original_name="a", hashed_name="a", is_augmented=True),
                        BaseColumnMetadata(original_name="b", hashed_name="b", is_augmented=True),
                        BaseColumnMetadata(original_name="c", hashed_name="c", is_augmented=True),
                    ],
                ),
            ],
        )
    ]

    assert enricher.get_autofe_features_description() is None
    assert enricher._has_single_ensemble_score() is False


def test_shap_fill_class_maps_provider_source():
    data = assemble_report_data(
        search_id="search-abc",
        search_keys=["PHONE"],
        bundle=bundle,
        features=[
            FeatureRow(name="client_feat", shap=0.7, provider="", source=CLIENT_SOURCE),
            FeatureRow(name="autofe_feat", shap=0.2, provider="Upgini", source=GENERATED_SOURCE),
            FeatureRow(name="upgini_feat", shap=0.056, provider="Upgini", source="Usage Data"),
            FeatureRow(name="ext_feat", shap=0.04, provider="Experian", source="Credit Bureau"),
        ],
    )
    html = generate_html_report(data)
    payload = _parse_report_data(html)
    by_name = {row["name"]: row for row in payload["features"]}

    assert by_name["client_feat"]["shapClass"] == "user"
    assert by_name["client_feat"]["provider"] == CLIENT_SOURCE
    assert by_name["autofe_feat"]["shapClass"] == "autofe"
    assert by_name["autofe_feat"]["provider"] == "Upgini"
    assert by_name["upgini_feat"]["shapClass"] == "upgini"
    assert by_name["upgini_feat"]["provider"] == "Upgini"
    assert by_name["ext_feat"]["shapClass"] == "external"
    assert by_name["ext_feat"]["provider"] == "Experian"
    assert ".shap-fill.external{background:#7b5cff}" in html
    assert "f.shapClass || 'upgini'" in html
    assert "providerBadge" in html
    assert "replace(/,\\s*/g, '<br>')" in html


def test_shap_layout_keeps_long_feature_names_visible():
    long_name = "f_autofe_div_" + "very_long_component_" * 8 + "end"
    data = assemble_report_data(
        search_id="search-abc",
        search_keys=["PHONE"],
        bundle=bundle,
        features=[
            FeatureRow(name=long_name, shap=0.4, provider="Upgini", source=GENERATED_SOURCE),
            FeatureRow(name="short", shap=0.2, provider="", source=CLIENT_SOURCE),
        ],
    )
    html = generate_html_report(data)
    payload = _parse_report_data(html)

    assert payload["features"][0]["name"] == long_name
    assert 'class="shap-name"' in html
    assert 'class="shap-value"' in html
    assert "grid-template-columns:fit-content(55%) minmax(90px,1fr) 64px" in html
    assert "grid-template-columns:subgrid" in html
    assert "overflow-wrap:anywhere" in html
    assert "f.importance / maxImportance * 100" in html
    assert "minmax(170px,1fr) 2fr 60px" not in html


def test_provider_badge_keeps_comma_separated_providers_in_payload():
    data = assemble_report_data(
        search_id="search-abc",
        search_keys=["PHONE"],
        bundle=bundle,
        features=[
            FeatureRow(
                name="f_autofe_sim_jw2",
                shap=0.03,
                provider="Training dataset, IP2Location",
                source=GENERATED_SOURCE,
            ),
        ],
    )
    html = generate_html_report(data)
    payload = _parse_report_data(html)
    assert payload["features"][0]["provider"] == "Training dataset, IP2Location"


def test_generated_source_keeps_text_after_comma():
    source = "AutoFE: features from Training dataset,Company IP Address Data"
    features, sources, _, _ = build_search_results(
        features_meta=[
            FeaturesMetadataV2(
                name="f_autofe_sim_jw2",
                type="numeric",
                source="generated",
                hit_rate=85.0,
                shap_value=0.062,
                data_provider="Training dataset",
                data_source=source,
            )
        ],
        is_ensemble=lambda name: False,
    )
    html = generate_html_report(
        assemble_report_data(search_id="search-abc", search_keys=["PHONE"], bundle=bundle, features=features, sources=sources)
    )
    payload = _parse_report_data(html)

    assert features[0].source == source
    assert payload["features"][0]["source"] == source
    assert payload["features"][0]["shapClass"] == "autofe"
    assert payload["searchResults"]["sources"][0]["source"] == source


def test_overview_stability_dashes_when_psi_missing():
    _, _, _, summary = build_search_results(
        features_meta=[
            FeaturesMetadataV2(name="client", type="numeric", source="etalon", hit_rate=100.0, shap_value=0.5),
            FeaturesMetadataV2(
                name="ads_feat",
                type="numeric",
                source="ads",
                hit_rate=100.0,
                shap_value=0.2,
                data_provider="Upgini",
                data_source="Usage Data",
            ),
        ],
        is_ensemble=lambda name: False,
        generated_features=[
            GeneratedFeatureMetadata(
                formula="catboost_score(client,ads_feat)",
                display_index="fold1",
                base_columns=[
                    BaseColumnMetadata(original_name="client", hashed_name="client", is_augmented=False),
                    BaseColumnMetadata(
                        original_name="ads_feat",
                        hashed_name="ads_feat",
                        ads_definition_id="ads-1",
                        is_augmented=False,
                    ),
                ],
            )
        ],
        joined_ads_features_count=10,
        joined_ads_count=2,
    )
    html = generate_html_report(
        assemble_report_data(search_id="search-abc", search_keys=["PHONE"], bundle=bundle, summary=summary)
    )
    payload = _parse_report_data(html)

    assert summary.stable_features_share is None
    assert payload["summaryCards"][0] == {"label": "Features found", "value": "10", "caption": ""}
    assert payload["summaryCards"][1] == {
        "label": "Used in model",
        "value": "2",
        "caption": "1 external · 1 original · 0 AutoFE",
    }
    assert payload["summaryCards"][2] == {"label": "Data sources", "value": "2", "caption": "1 contributed"}
    assert payload["summaryCards"][3] == {"label": "Stability", "value": "—", "caption": ""}


def test_overview_contributed_counts_nested_ensemble_leaf_ads():
    _, _, _, summary = build_search_results(
        features_meta=[
            FeaturesMetadataV2(
                name="ads_a",
                type="numeric",
                source="ads",
                hit_rate=100.0,
                shap_value=0.2,
                data_provider="Upgini",
                data_source="Markets data",
            ),
            FeaturesMetadataV2(
                name="f_autofe_div",
                type="numeric",
                source="generated",
                hit_rate=100.0,
                shap_value=0.1,
                data_provider="Upgini",
                data_source="AutoFE: features from POI data OpenStreetMap",
            ),
            FeaturesMetadataV2(name="client", type="numeric", source="etalon", hit_rate=100.0, shap_value=0.5),
            FeaturesMetadataV2(
                name="f_autofe_ensemble",
                type="numeric",
                source="generated",
                hit_rate=100.0,
                shap_value=1.0,
                data_provider="Upgini",
                data_source="AutoFE: features from Markets data",
            ),
        ],
        is_ensemble=lambda name: name == "f_autofe_ensemble",
        generated_features=[
            GeneratedFeatureMetadata(
                formula="ensemble_score(catboost_score_fold_0(ads_a,(ads_b/ads_c),client),catboost_score_fold_1(ads_a,client))",
                display_index="ens",
                base_columns=[
                    BaseColumnMetadata(
                        original_name="ads_a",
                        hashed_name="ads_a",
                        ads_definition_id="ads-1",
                        is_augmented=False,
                    ),
                    BaseColumnMetadata(
                        original_name="ads_b",
                        hashed_name="ads_b",
                        ads_definition_id="ads-2",
                        is_augmented=False,
                    ),
                    BaseColumnMetadata(
                        original_name="ads_c",
                        hashed_name="ads_c",
                        ads_definition_id="ads-3",
                        is_augmented=False,
                    ),
                    BaseColumnMetadata(original_name="client", hashed_name="client", is_augmented=False),
                ],
            )
        ],
        joined_ads_count=85,
    )
    html = generate_html_report(
        assemble_report_data(search_id="search-abc", search_keys=["PHONE"], bundle=bundle, summary=summary)
    )
    payload = _parse_report_data(html)

    assert summary.contributed_sources == 3
    assert payload["summaryCards"][2] == {"label": "Data sources", "value": "85", "caption": "3 contributed"}


def test_overview_contributed_ignores_wrapper_ensemble_unused_ads():
    _, _, _, summary = build_search_results(
        features_meta=[
            FeaturesMetadataV2(
                name="ads_feat",
                type="numeric",
                source="ads",
                hit_rate=100.0,
                shap_value=0.2,
                data_provider="Upgini",
                data_source="Usage Data",
            ),
            FeaturesMetadataV2(name="client", type="numeric", source="etalon", hit_rate=100.0, shap_value=0.5),
        ],
        is_ensemble=lambda name: False,
        generated_features=[
            GeneratedFeatureMetadata(
                formula="ensemble_score(model1,model2)",
                display_index="ens",
                base_columns=[
                    BaseColumnMetadata(
                        original_name="unused_ads",
                        hashed_name="unused_ads",
                        ads_definition_id="ads-unused",
                        is_augmented=False,
                    ),
                ],
            ),
            GeneratedFeatureMetadata(
                formula="catboost_score(client,ads_feat)",
                display_index="fold1",
                base_columns=[
                    BaseColumnMetadata(original_name="client", hashed_name="client", is_augmented=False),
                    BaseColumnMetadata(
                        original_name="ads_feat",
                        hashed_name="ads_feat",
                        ads_definition_id="ads-1",
                        is_augmented=False,
                    ),
                ],
            ),
        ],
        joined_ads_count=17,
    )

    assert summary.contributed_sources == 1


def test_overview_stability_ignores_empty_psi():
    _, _, _, summary = build_search_results(
        features_meta=[
            FeaturesMetadataV2(
                name="stable", type="numeric", source="etalon", hit_rate=100.0, shap_value=0.4, psi_value=0.0
            ),
            FeaturesMetadataV2(name="missing_psi", type="numeric", source="etalon", hit_rate=100.0, shap_value=0.3),
            FeaturesMetadataV2(
                name="unstable",
                type="numeric",
                source="ads",
                hit_rate=100.0,
                shap_value=0.2,
                data_provider="Upgini",
                data_source="Usage Data",
                psi_value=0.3,
            ),
        ],
        is_ensemble=lambda name: False,
        generated_features=[
            GeneratedFeatureMetadata(
                formula="catboost_score(stable,missing_psi,unstable)",
                display_index="fold1",
                base_columns=[
                    BaseColumnMetadata(original_name="stable", hashed_name="stable", is_augmented=False),
                    BaseColumnMetadata(original_name="missing_psi", hashed_name="missing_psi", is_augmented=False),
                    BaseColumnMetadata(
                        original_name="unstable",
                        hashed_name="unstable",
                        ads_definition_id="ads-1",
                        is_augmented=False,
                    ),
                ],
            )
        ],
    )
    html = generate_html_report(
        assemble_report_data(search_id="search-abc", search_keys=["PHONE"], bundle=bundle, summary=summary)
    )
    payload = _parse_report_data(html)

    assert summary.psi_features == 2
    assert summary.stable_features == 1
    assert summary.stable_features_share == 0.5
    assert payload["summaryCards"][3] == {"label": "Stability", "value": "50%", "caption": "1 of 2 are stable"}


def test_overview_caption_from_nested_ensemble_formula():
    _, _, _, summary = build_search_results(
        features_meta=[
            FeaturesMetadataV2(
                name="f_autofe_ensemble_x_abc",
                type="numeric",
                source="generated",
                hit_rate=100.0,
                shap_value=1.0,
            ),
            FeaturesMetadataV2(
                name="f_model1_abc",
                type="numeric",
                source="ads",
                hit_rate=100.0,
                shap_value=0.1,
                data_provider="Upgini",
                data_source="Usage Data",
            ),
            FeaturesMetadataV2(
                name="pd002_6e6a41", type="numeric", source="etalon", hit_rate=100.0, shap_value=0.7
            ),
        ],
        is_ensemble=lambda name: "ensemble" in name,
        generated_features=[
            GeneratedFeatureMetadata(
                formula=(
                    "ensemble_x("
                    "catboost_x_fold_0(f_model1_abc,pd002_6e6a41,sim_jw1(combined_bio,form_car),"
                    "sim_jw1(combined_bio,form_children)),"
                    "catboost_x_fold_1(f_model1_abc,pd002_6e6a41,sim_jw1(combined_bio,form_car),"
                    "sim_jw1(combined_bio,form_children)))"
                ),
                display_index="abc",
                base_columns=[
                    BaseColumnMetadata(
                        original_name="f_model1_abc",
                        hashed_name="f_model1_abc",
                        ads_definition_id="ads-1",
                        is_augmented=False,
                    ),
                    BaseColumnMetadata(
                        original_name="pd002_6e6a41", hashed_name="pd002_6e6a41", is_augmented=False
                    ),
                    BaseColumnMetadata(
                        original_name="combined_bio", hashed_name="combined_bio", is_augmented=False
                    ),
                    BaseColumnMetadata(
                        original_name="form_car", hashed_name="form_car", is_augmented=False
                    ),
                    BaseColumnMetadata(
                        original_name="form_children", hashed_name="form_children", is_augmented=False
                    ),
                ],
            )
        ],
        joined_ads_features_count=10,
        joined_ads_count=3,
    )
    html = generate_html_report(
        assemble_report_data(search_id="search-abc", search_keys=["PHONE"], bundle=bundle, summary=summary)
    )
    payload = _parse_report_data(html)

    assert summary.model_features == 2
    assert summary.external_features == 1
    assert summary.original_features == 1
    assert summary.autofe_features == 0
    assert payload["summaryCards"][1] == {
        "label": "Used in model",
        "value": "2",
        "caption": "1 external · 1 original · 0 AutoFE",
    }


def test_report_button_downloads_html_like_pdf():
    from upgini.utils.display_utils import _html_action_button, _report_button_html

    html = _report_button_html("<html>ok</html>", download_name="upgini-report-search-abc.html")
    payload = base64.b64encode(b"<html>ok</html>").decode()
    assert html == _html_action_button(
        "Open full report",
        f"data:text/html;base64,{payload}",
        download_name="upgini-report-search-abc.html",
    )
    assert 'download="upgini-report-search-abc.html"' in html
    assert "<button>Open full report</button>" in html
