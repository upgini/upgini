from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from upgini.report.data import FeatureRow, ReportCharts, ReportData, SampleStats, SearchResultsSummary, SourceRow
from upgini.report.stats import MEAN_AXIS_PADDING
from upgini.utils.feature_info import CLIENT_SOURCE, GENERATED_SOURCE

_TEMPLATE_PATH = Path(__file__).with_name("template.html")
_SAMPLE_COLORS = ("#1645ee", "#20aab4", "#53cc61", "#e19132", "#d45b5b")


def generate_html_report(data: ReportData) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if "__REPORT_DATA__" not in template:
        raise RuntimeError("Report template is missing the REPORT_DATA placeholder")
    payload = json.dumps(_report_payload(data), ensure_ascii=False, indent=2)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return template.replace("__REPORT_DATA__", payload, 1)


def _report_payload(data: ReportData) -> dict:
    samples = _samples(data)
    charts = data.charts or ReportCharts()
    return {
        "meta": _meta(data, samples),
        "timeline": {
            "months": list(charts.timeline_months),
            "periodOptions": list(charts.period_options),
            "defaultPeriod": charts.default_period,
        },
        "samples": samples,
        "keyResult": _key_result(data, charts),
        "summaryCards": _summary_cards(data.summary),
        "sampleStats": _sample_stats_payload(data, samples, charts),
        "scoreAnalysis": _score_analysis_payload(charts, samples),
        "scoreDistribution": _score_distribution_payload(charts, samples),
        "scoreStability": _score_stability_payload(charts, samples),
        "features": _features_payload(data.features),
        "shap": {"topN": 5},
        "searchResults": {
            "relevantFeaturesCount": _dash(len(data.features) if data.features else data.summary.relevant_features),
            "dataSourcesCount": _dash(data.summary.data_sources),
            "autofeCount": _dash(len(data.autofe)),
            "sources": _sources_payload(data.sources),
            "autofe": list(data.autofe),
        },
    }


def _meta(data: ReportData, samples: list[dict]) -> dict:
    meta = data.metadata
    payload = {
        "reportTitle": "Feature enrichment report",
        "reportSubtitle": (
            "Impact of external data on model quality, score dynamics, and the stability of discovered features."
        ),
        "searchId": meta.search_id,
        "generatedAt": meta.generated_at,
        "pylibVersion": meta.pylib_version,
        "searchKeys": list(meta.search_keys),
        "sampleNames": [sample["id"] for sample in samples],
        "searchDuration": meta.search_duration,
        "sourcesCount": data.summary.data_sources,
        "totalRows": meta.reference_rows,
        "statusLine": "FIT completed successfully",
        "downloadFileName": f"upgini-report-{meta.search_id}.html",
    }
    if meta.logo_url:
        payload["partnerLogo"] = {"src": meta.logo_url, "alt": "", "height": 30}
    return payload


def _samples(data: ReportData) -> list[dict]:
    names = [stats.sample for stats in data.sample_stats] or list(data.metadata.samples)
    return [
        {
            "id": _slug(name),
            "label": name,
            "caption": "OOF validation" if index == 0 else "",
            "color": _SAMPLE_COLORS[index % len(_SAMPLE_COLORS)],
        }
        for index, name in enumerate(names)
    ]


def _key_result(data: ReportData, charts: ReportCharts) -> dict:
    series = {
        (_slug(row.metric or "score"), _slug(row.evaluation_scope)): row for row in charts.quality_monthly
    }
    metrics: dict[str, dict] = {}
    for sample in data.quality_by_sample:
        key = _slug(sample.metric or "score")
        bucket = metrics.setdefault(key, {"name": sample.metric or "score", "bySample": {}})
        monthly = series.get((key, _slug(sample.evaluation_scope)))
        bucket["bySample"][_slug(sample.evaluation_scope)] = {
            "baseline": sample.baseline,
            "baselineCi": sample.baseline_std,
            "enriched": sample.enriched,
            "enrichedCi": sample.enriched_std,
            "baselineSeries": list(monthly.baseline) if monthly else [],
            "enrichedSeries": list(monthly.enriched) if monthly else [],
        }
    for monthly in charts.quality_monthly:
        key = _slug(monthly.metric or "score")
        sample_id = _slug(monthly.evaluation_scope)
        bucket = metrics.setdefault(key, {"name": monthly.metric or "score", "bySample": {}})
        if sample_id in bucket["bySample"]:
            continue
        bucket["bySample"][sample_id] = {
            "baseline": None,
            "baselineCi": None,
            "enriched": None,
            "enrichedCi": None,
            "baselineSeries": list(monthly.baseline),
            "enrichedSeries": list(monthly.enriched),
        }
    default_metric = next(iter(metrics), "")
    has_std = any(
        sample.baseline_std is not None or sample.enriched_std is not None for sample in data.quality_by_sample
    )
    return {
        "title": "Key result",
        "headline": "",
        "defaultMetric": default_metric,
        "confidenceNote": "± values are cross-validation standard deviation" if has_std else "",
        "columns": {
            "sample": "Sample",
            "baseline": "Baseline",
            "enriched": "Enriched",
            "upliftAbs": "Uplift, abs",
            "upliftPct": "Uplift, %",
        },
        "metrics": metrics,
    }


def _summary_cards(summary: SearchResultsSummary) -> list[dict]:
    stability = None
    if summary.stable_features_share is not None:
        stability = f"{summary.stable_features_share * 100:.0f}%"
    sources_caption = ""
    if summary.contributed_sources is not None:
        sources_caption = f"{summary.contributed_sources} contributed"
    stability_caption = ""
    if summary.stable_features is not None and summary.psi_features is not None:
        stability_caption = f"{summary.stable_features} of {summary.psi_features} are stable"
    features_found = summary.features_found if summary.features_found is not None else summary.relevant_features
    data_sources = summary.joined_sources if summary.joined_sources is not None else summary.data_sources
    return [
        {"label": "Features found", "value": _dash(features_found), "caption": ""},
        {"label": "Used in model", "value": _dash(summary.model_features), "caption": _used_in_model_caption(summary)},
        {"label": "Data sources", "value": _dash(data_sources), "caption": sources_caption},
        {"label": "Stability", "value": _dash(stability), "caption": stability_caption},
    ]


def _used_in_model_caption(summary: SearchResultsSummary) -> str:
    if summary.external_features is None and summary.original_features is None and summary.autofe_features is None:
        return ""
    return " · ".join(
        [
            f"{summary.external_features or 0} external",
            f"{summary.original_features or 0} original",
            f"{summary.autofe_features or 0} AutoFE",
        ]
    )


def _features_payload(features: list[FeatureRow]) -> list[dict]:
    rows = []
    for row in features:
        shap_class = _shap_fill_class(row)
        provider = row.provider
        if not provider and shap_class == "user":
            provider = CLIENT_SOURCE
        rows.append(
            {
                "name": row.name,
                "provider": provider,
                "source": row.source,
                "shapClass": shap_class,
                "importance": abs(row.shap) if row.shap is not None else 0,
                "shap": row.shap,
                "coverage": row.coverage,
                "status": row.stability_status,
                "psi": row.psi,
                "drift": row.drift,
            }
        )
    return rows


def _shap_fill_class(row: FeatureRow) -> str:
    source = (row.source or "").lower()
    provider = (row.provider or "").lower()
    if source.startswith("autofe") or provider == "autofe" or source == GENERATED_SOURCE.lower():
        return "autofe"
    if source == CLIENT_SOURCE.lower() or provider == "user":
        return "user"
    if provider in {"", "upgini"}:
        return "upgini"
    return "external"


def _sources_payload(sources: list[SourceRow]) -> list[dict]:
    return [
        {
            "provider": row.provider,
            "source": row.source,
            "aggregateShap": row.shap_sum,
            "relevantFeatures": row.feature_count,
        }
        for row in sources
    ]


def _sample_stats_payload(data: ReportData, samples: list[dict], charts: ReportCharts) -> dict:
    by_id = {_slug(stats.sample): stats for stats in data.sample_stats}
    rows = [
        _stat_row("Date range", samples, by_id, lambda stats: stats.date_range),
        _stat_row("Count", samples, by_id, lambda stats: _format_int(stats.rows)),
        _stat_row("Count (target)", samples, by_id, lambda stats: _format_int(stats.labeled)),
    ]
    if data.is_binary:
        rows.append(_stat_row("Count 1's (target)", samples, by_id, lambda stats: _format_int(stats.positive)))
    rows.extend(
        [
            _stat_row("Mean (target)", samples, by_id, lambda stats: _format_mean(stats.mean_target)),
            _stat_row("Count (unlabeled)", samples, by_id, lambda stats: _format_int(stats.unlabeled)),
        ]
    )
    return {
        "rows": rows,
        "monthlyAxis": list(charts.timeline_months),
        "monthlyCountMax": charts.monthly_count_max or 1,
        "monthlyMeanAxisPadding": MEAN_AXIS_PADDING,
        "monthlyPoints": [
            {
                "month": point.month,
                "sample": _slug(point.sample),
                "total": point.total,
                "labeled": point.labeled,
                "unlabeled": point.unlabeled,
                "positive": point.positive,
                "mean": point.mean,
            }
            for point in charts.sample_monthly
        ],
    }


def _score_analysis_payload(charts: ReportCharts, samples: list[dict]) -> dict:
    by_id = {_slug(row.sample): row for row in charts.score_monthly}
    return {
        "countAxisMax": charts.score_count_axis_max or 1,
        "bySample": {
            sample["id"]: {
                "meanScore": list(row.mean_score),
                "meanTarget": list(row.mean_target),
                "labeled": list(row.labeled),
                "unlabeled": list(row.unlabeled),
            }
            for sample in samples
            if (row := by_id.get(sample["id"])) is not None
        },
    }


def _score_distribution_payload(charts: ReportCharts, samples: list[dict]) -> dict:
    by_id = {_slug(row.sample): row for row in charts.histograms}
    train = next((by_id.get(sample["id"]) for sample in samples if sample["id"] in by_id), None)
    return {
        "binEdges": list(charts.histogram_bin_edges),
        "densityAxisMax": charts.histogram_density_max or 1,
        "countScale": {
            "target0": train.n_target_0 if train else 0,
            "target1": train.n_target_1 if train else 0,
        },
        "bySample": {
            sample["id"]: {"target0": list(row.target_0), "target1": list(row.target_1)}
            for sample in samples
            if (row := by_id.get(sample["id"])) is not None
        },
    }


def _score_stability_payload(charts: ReportCharts, samples: list[dict]) -> dict:
    by_id = {_slug(row.sample): row for row in charts.score_psi}
    return {
        "psiAxisMax": charts.psi_axis_max or charts.psi_critical,
        "thresholds": {"warning": charts.psi_warning, "critical": charts.psi_critical},
        "bySample": {
            sample["id"]: {
                "psi": list(row.psi),
                "rows": list(row.rows),
                "scoreCoverage": list(row.coverage),
            }
            for sample in samples
            if (row := by_id.get(sample["id"])) is not None
        },
    }


def _stat_row(label: str, samples: list[dict], by_id: dict[str, SampleStats], value) -> dict:
    values = {}
    for sample in samples:
        stats = by_id.get(sample["id"]) or SampleStats(sample=sample["label"])
        values[sample["id"]] = _dash(value(stats))
    return {"label": label, "values": values}


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _dash(value: Optional[object]) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def _format_int(value: Optional[int]) -> Optional[str]:
    return f"{value:,}" if value is not None else None


def _format_mean(value: Optional[float]) -> Optional[str]:
    return f"{value:.3f}" if value is not None else None
