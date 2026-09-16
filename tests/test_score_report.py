import base64
from pathlib import Path

import pandas as pd
from requests_mock.mocker import Mocker

from upgini.features_enricher import FeaturesEnricher
from upgini.metadata import (
    BaseColumnMetadata,
    FeaturesMetadataV2,
    GeneratedFeatureMetadata,
    ProviderTaskMetadataV2,
    SearchKey,
)
from upgini.report.assemble import assemble_report_data, format_search_duration
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


def test_format_search_duration():
    assert format_search_duration(None) is None
    assert format_search_duration(9) == "9 sec"
    assert format_search_duration(62) == "1 min 2 sec"
    assert format_search_duration(3723) == "1 h 2 min 3 sec"


def test_generate_html_report_from_assembled_data():
    metrics_df = pd.DataFrame(
        {
            bundle.get("quality_metrics_segment_header"): ["Train", "Eval 1"],
            bundle.get("quality_metrics_rows_header"): [100, 40],
            bundle.get("quality_metrics_mean_target_header"): [0.31, 0.28],
            bundle.get("quality_metrics_baseline_header").format("GINI"): ["0.400", "0.390"],
            bundle.get("quality_metrics_enriched_header").format("GINI"): ["0.512", "0.480"],
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
        bundle=bundle,
    )
    html = generate_html_report(data)

    assert "search-abc" in html
    assert "PHONE, DATE" in html
    assert "2 min 5 sec" in html
    assert "0.512" in html
    assert "Features found" not in html
    assert "f_autofe_ensemble_score_abc123" not in html
    assert data.quality_by_sample[0].metric == "GINI"
    assert data.quality_by_sample[0].enriched == "0.512"
    assert data.features == []
    assert data.sources == []
    assert data.autofe == []
    assert data.summary.relevant_features is None
    assert data.summary.model_features is None


def _ensemble_enricher(url: str, tmp_path: Path, ensemble_col: str = "f_autofe_upgini_score_abc123") -> FeaturesEnricher:
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        logs_enabled=False,
        reports_path=str(tmp_path),
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


def test_write_score_report_to_custom_path(requests_mock: Mocker, tmp_path: Path):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    enricher = _ensemble_enricher(url, tmp_path)

    path = enricher._write_score_report()

    assert path == str(tmp_path / "upgini-report-search-abc.html")
    html = Path(path).read_text(encoding="utf-8")
    assert "search-abc" in html
    assert "0.610" in html
    assert "42 sec" in html


def test_html_report_button_does_not_build_pdf(requests_mock: Mocker, tmp_path: Path, monkeypatch):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    enricher = _ensemble_enricher(url, tmp_path)
    pdf_calls = []
    monkeypatch.setattr("upgini.features_enricher.ipython_available", lambda: True)
    monkeypatch.setattr("upgini.features_enricher.show_button_open_report", lambda *args, **kwargs: "html")
    monkeypatch.setattr(
        "upgini.features_enricher.prepare_and_show_report",
        lambda *args, **kwargs: pdf_calls.append(True) or "pdf",
    )

    assert enricher._FeaturesEnricher__show_report_button() == "html"
    assert pdf_calls == []
    assert (tmp_path / "upgini-report-search-abc.html").exists()
    assert list(tmp_path.glob("*.pdf")) == []


def test_write_score_report_skipped_without_ensemble(requests_mock: Mocker, tmp_path: Path):
    url = "https://some.fake.url"
    mock_default_requests(requests_mock, url)
    enricher = FeaturesEnricher(endpoint=url, logs_enabled=False, reports_path=str(tmp_path))
    enricher._search_task = SearchTask("search-abc")
    enricher.feature_names_ = ["ads_feature"]

    assert enricher._write_score_report() is None
    assert list(tmp_path.iterdir()) == []


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


def test_report_button_matches_pdf_markup_and_downloads_on_hosted_notebook(tmp_path, monkeypatch):
    from upgini.utils.display_utils import _html_action_button, _report_button_html

    report = tmp_path / "upgini-report-search-abc.html"
    report.write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr("upgini.utils.track_info.is_hosted_notebook", lambda: True)

    html = _report_button_html(str(report))
    payload = base64.b64encode(b"<html>ok</html>").decode()
    assert html == _html_action_button(
        "Open full report",
        f"data:text/html;base64,{payload}",
        download_name="upgini-report-search-abc.html",
    )
    assert 'download="upgini-report-search-abc.html"' in html
    assert "<button>Open full report</button>" in html


def test_report_button_opens_local_file(tmp_path, monkeypatch):
    from upgini.utils.display_utils import _html_action_button, _report_button_html

    report = tmp_path / "upgini-report-search-abc.html"
    report.write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr("upgini.utils.track_info.is_hosted_notebook", lambda: False)

    html = _report_button_html(str(report))
    assert html == _html_action_button("Open full report", report.resolve().as_uri())
    assert "download=" not in html
