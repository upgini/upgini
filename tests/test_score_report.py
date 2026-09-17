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
from upgini.report.assemble import UPGINI_REPORT_BRANDING_URL, assemble_report_data, format_search_duration
from upgini.report.html import generate_html_report
from upgini.resource_bundle import bundle
from upgini.search_task import SearchTask

from .utils import mock_default_requests

SearchTask.PROTECT_FROM_RATE_LIMIT = False


def _ensemble_metadata(ensemble_col: str) -> ProviderTaskMetadataV2:
    return ProviderTaskMetadataV2(
        features=[
            FeaturesMetadataV2(name=ensemble_col, type="numeric", source="ads", hit_rate=100.0, shap_value=1.0)
        ],
        generated_features=[
            GeneratedFeatureMetadata(
                alias="upgini_score",
                formula="ensemble_score(model1,model2)",
                display_index="abc123",
                base_columns=[
                    BaseColumnMetadata(original_name="model1", hashed_name="model1", is_augmented=False),
                    BaseColumnMetadata(original_name="model2", hashed_name="model2", is_augmented=False),
                ],
            )
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
    assert payload["summaryCards"][1] == {"label": "Used in model", "value": "2", "caption": ""}
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
    assert "Features SHAP" in html
    assert "Search results" in html
    assert "Feature stability" in html
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

    assert html is not None
    assert "search-abc" in html
    assert '"PHONE"' in html
    assert "SearchKey" not in html
    assert '"enriched": 0.61' in html
    assert "42 sec" in html
    assert '"label": "Used in model"' in html
    assert '"value": "2"' in html
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
