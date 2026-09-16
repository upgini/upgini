from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from upgini.report.data import ReportData, SampleStats, SearchResultsSummary

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
    return {
        "meta": _meta(data, samples),
        "timeline": {"months": [], "periodOptions": [], "defaultPeriod": 0},
        "samples": samples,
        "keyResult": _key_result(data),
        "summaryCards": _summary_cards(data.summary),
        "sampleStats": _sample_stats_payload(data, samples),
        "scoreAnalysis": {"bySample": {}},
        "scoreDistribution": {"bySample": {}},
        "scoreStability": {"bySample": {}},
        "features": [],
        "featureStability": {"bySample": {}},
        "shap": {"topN": 5},
        "searchResults": {
            "relevantFeaturesCount": _dash(data.summary.relevant_features),
            "dataSourcesCount": _dash(data.summary.data_sources),
            "autofeCount": _dash(len(data.autofe) if data.autofe else None),
            "sources": [],
            "autofe": [],
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
            "caption": "",
            "color": _SAMPLE_COLORS[index % len(_SAMPLE_COLORS)],
        }
        for index, name in enumerate(names)
    ]


def _key_result(data: ReportData) -> dict:
    metrics: dict[str, dict] = {}
    for sample in data.quality_by_sample:
        key = _slug(sample.metric or "score")
        bucket = metrics.setdefault(key, {"name": sample.metric or "score", "bySample": {}})
        bucket["bySample"][_slug(sample.evaluation_scope)] = {
            "baseline": sample.baseline,
            "baselineCi": sample.baseline_std,
            "enriched": sample.enriched,
            "enrichedCi": sample.enriched_std,
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
    return [
        {"label": "Features found", "value": _dash(summary.relevant_features), "caption": ""},
        {"label": "Used in model", "value": _dash(summary.model_features), "caption": ""},
        {"label": "Data sources", "value": _dash(summary.data_sources), "caption": ""},
        {"label": "Stability", "value": _dash(stability), "caption": ""},
    ]


def _sample_stats_payload(data: ReportData, samples: list[dict]) -> dict:
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
    return {"rows": rows, "monthlyAxis": [], "monthlyPoints": []}


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
