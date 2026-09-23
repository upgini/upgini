from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from upgini.__about__ import __version__
from upgini.autofe.feature import Column, Feature
from upgini.autofe.vector import CatboostModel, EnsembleModel, OnnxModel
from upgini.metadata import BaseColumnMetadata, FeaturesMetadataV2, GeneratedFeatureMetadata
from upgini.report.data import (
    FeatureRow,
    ModelFeatureShap,
    QualitySample,
    ReportData,
    ReportMetadata,
    SampleStats,
    SearchResultsSummary,
    SourceRow,
)
from upgini.report.stats import PSI_CRITICAL, PSI_WARNING, compute_report_charts
from upgini.resource_bundle import ResourceBundle
from upgini.utils.feature_info import CLIENT_SOURCE, GENERATED_SOURCE, FeatureInfo

UPGINI_REPORT_BRANDING_URL = "UPGINI_REPORT_BRANDING_URL"
_STABILITY_PSI = 0.2


def format_search_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min {secs} sec"
    if minutes:
        return f"{minutes} min {secs} sec"
    return f"{secs} sec"


def assemble_report_data(
    *,
    search_id: str,
    search_keys: list[str],
    generated_at: Optional[datetime] = None,
    pylib_version: Optional[str] = None,
    search_duration_seconds: Optional[float] = None,
    reference_rows: Optional[int] = None,
    samples: Optional[list[str]] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    metric_name: Optional[str] = None,
    model_features: Optional[int] = None,
    is_binary: bool = False,
    dataset_samples: Optional[dict[str, pd.DataFrame]] = None,
    scored_samples: Optional[dict[str, pd.DataFrame]] = None,
    bundle: ResourceBundle,
    summary: Optional[SearchResultsSummary] = None,
    features: Optional[list[FeatureRow]] = None,
    sources: Optional[list[SourceRow]] = None,
    autofe: Optional[list[dict[str, str]]] = None,
    model_feature_shap: Optional[list[ModelFeatureShap]] = None,
) -> ReportData:
    generated_at = generated_at or datetime.now(timezone.utc)
    sample_names = list(samples or [])
    chart_stats, charts = compute_report_charts(
        sample_names=sample_names,
        dataset_samples=dataset_samples,
        scored_samples=scored_samples,
        metric_name=metric_name,
        is_binary=is_binary,
    )
    summary = summary or SearchResultsSummary(model_features=model_features)
    if summary.model_features is None:
        summary.model_features = model_features
    return ReportData(
        metadata=ReportMetadata(
            search_id=search_id,
            generated_at=generated_at.strftime("%Y-%m-%d %H:%M UTC"),
            pylib_version=pylib_version or __version__,
            search_keys=list(search_keys or []),
            search_duration=format_search_duration(search_duration_seconds),
            reference_rows=reference_rows,
            samples=sample_names,
            logo_url=_branding_url(),
        ),
        quality_by_sample=_quality_samples(metrics_df, metric_name, bundle),
        summary=summary,
        sample_stats=chart_stats or _sample_stats(sample_names, metrics_df, bundle),
        is_binary=is_binary,
        features=list(features or []),
        sources=list(sources or []),
        autofe=list(autofe or []),
        model_feature_shap=list(model_feature_shap or []),
        charts=charts,
    )


def build_search_results(
    *,
    features_meta: list[FeaturesMetadataV2],
    is_ensemble: Callable[[str], bool],
    generated_names: Optional[set[str]] = None,
    generated_features: Optional[list[GeneratedFeatureMetadata]] = None,
    joined_ads_features_count: Optional[int] = None,
    joined_ads_count: Optional[int] = None,
) -> tuple[list[FeatureRow], list[SourceRow], list[ModelFeatureShap], SearchResultsSummary]:
    generated_names = generated_names or set()
    model_rows: list[FeatureRow] = []
    found_rows: list[FeatureRow] = []
    for meta in features_meta:
        if is_ensemble(meta.name) or not _has_nonzero_shap_value(meta.shap_value):
            continue
        row = _feature_row(meta.name, meta, generated_names)
        model_rows.append(row)
        if meta.source != "etalon":
            found_rows.append(row)
    model_rows.sort(key=lambda row: (row.shap is None, -abs(row.shap or 0.0), row.name))
    model_shap = [
        ModelFeatureShap(
            rank=index,
            feature=row.name,
            provider=row.provider,
            mean_abs_shap=None if row.shap is None else abs(row.shap),
        )
        for index, row in enumerate(model_rows, start=1)
    ]
    sources = _source_rows(found_rows)
    summary = _overview_summary(
        model_rows=model_rows,
        found_rows=found_rows,
        sources=sources,
        is_ensemble=is_ensemble,
        generated_features=generated_features or [],
        joined_ads_features_count=joined_ads_features_count,
        joined_ads_count=joined_ads_count,
    )
    return model_rows, sources, model_shap, summary


def autofe_rows_from_description(df: Optional[pd.DataFrame], bundle: ResourceBundle) -> list[dict[str, str]]:
    if df is None or df.empty:
        return []
    sources_col = bundle.get("autofe_descriptions_sources")
    name_col = bundle.get("autofe_descriptions_feature_name")
    func_col = bundle.get("autofe_descriptions_function")
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        source_features = []
        for index in (1, 2):
            col = bundle.get("autofe_descriptions_feature").format(index)
            value = _as_str(row[col]) if col in df.columns else None
            if value:
                source_features.append(value)
        rows.append(
            {
                "sources": (_as_str(row[sources_col]) or "") if sources_col in df.columns else "",
                "generatedFeature": (_as_str(row[name_col]) or "") if name_col in df.columns else "",
                "sourceFeatures": ", ".join(source_features),
                "functions": (_as_str(row[func_col]) or "") if func_col in df.columns else "",
            }
        )
    return rows


def _branding_url() -> Optional[str]:
    env_url = os.getenv(UPGINI_REPORT_BRANDING_URL)
    if env_url and env_url.strip():
        return env_url.strip()
    return None


def _quality_samples(
    metrics_df: Optional[pd.DataFrame], metric_name: Optional[str], bundle: ResourceBundle
) -> list[QualitySample]:
    if metrics_df is None or metrics_df.empty:
        return []
    segment_col = bundle.get("quality_metrics_segment_header")
    uplift_col = bundle.get("quality_metrics_uplift_header")
    uplift_pct_col = bundle.get("quality_metrics_uplift_perc_header")
    baseline_col = bundle.get("quality_metrics_baseline_header").format(metric_name) if metric_name else None
    enriched_col = bundle.get("quality_metrics_enriched_header").format(metric_name) if metric_name else None
    baseline_std_col = bundle.get("quality_metrics_baseline_std_header").format(metric_name) if metric_name else None
    enriched_std_col = bundle.get("quality_metrics_enriched_std_header").format(metric_name) if metric_name else None

    samples: list[QualitySample] = []
    for _, row in metrics_df.iterrows():
        samples.append(
            QualitySample(
                evaluation_scope=(_as_str(row[segment_col]) or "") if segment_col in metrics_df.columns else "",
                metric=metric_name or "",
                baseline=_as_float(row[baseline_col]) if baseline_col in metrics_df.columns else None,
                baseline_std=_as_float(row[baseline_std_col]) if baseline_std_col in metrics_df.columns else None,
                enriched=_as_float(row[enriched_col]) if enriched_col in metrics_df.columns else None,
                enriched_std=_as_float(row[enriched_std_col]) if enriched_std_col in metrics_df.columns else None,
                uplift=_as_float(row[uplift_col]) if uplift_col in metrics_df.columns else None,
                relative_uplift=_as_str(row[uplift_pct_col]) if uplift_pct_col in metrics_df.columns else None,
            )
        )
    return samples


def _sample_stats(
    sample_names: list[str], metrics_df: Optional[pd.DataFrame], bundle: ResourceBundle
) -> list[SampleStats]:
    segment_col = bundle.get("quality_metrics_segment_header")
    rows_col = bundle.get("quality_metrics_rows_header")
    mean_col = bundle.get("quality_metrics_mean_target_header")
    by_name: dict[str, SampleStats] = {}
    if metrics_df is not None and not metrics_df.empty and segment_col in metrics_df.columns:
        for _, row in metrics_df.iterrows():
            name = _as_str(row[segment_col])
            if not name:
                continue
            by_name[name] = SampleStats(
                sample=name,
                rows=_as_int(row[rows_col]) if rows_col in metrics_df.columns else None,
                mean_target=_as_float(row[mean_col]) if mean_col in metrics_df.columns else None,
            )
    names = sample_names or list(by_name)
    stats = [by_name.get(name) or SampleStats(sample=name) for name in names]
    seen = set(names)
    stats.extend(sample for name, sample in by_name.items() if name not in seen)
    return stats


def _as_str(value) -> Optional[str]:
    return None if pd.isna(value) else str(value)


def _as_float(value) -> Optional[float]:
    return None if pd.isna(value) else float(value)


def _as_int(value) -> Optional[int]:
    return None if pd.isna(value) else int(value)


def _feature_row(
    name: str, meta: Optional[FeaturesMetadataV2], generated_names: Optional[set[str]] = None
) -> FeatureRow:
    if meta is None:
        return FeatureRow(name=name)
    generated_names = generated_names or set()
    is_generated = name in generated_names or meta.source == "generated"
    is_client = meta.source == "etalon" and not is_generated
    info = FeatureInfo.from_metadata(meta, None, is_client, is_generated)
    return FeatureRow(
        name=name,
        shap=meta.shap_value,
        psi=meta.psi_value,
        drift=meta.drift_score,
        coverage=meta.hit_rate,
        provider=info.internal_provider,
        source=_feature_source(meta, is_generated, info.internal_source),
        stability_status=_psi_status(meta.psi_value),
    )


def _feature_source(meta: FeaturesMetadataV2, is_generated: bool, fallback: str) -> str:
    if is_generated:
        if meta.data_source:
            return meta.data_source
        if meta.data_sources:
            return ", ".join(meta.data_sources)
    return fallback


def _source_rows(rows: list[FeatureRow]) -> list[SourceRow]:
    grouped: dict[tuple[str, str], list[FeatureRow]] = {}
    for row in rows:
        if not row.provider:
            continue
        grouped.setdefault((row.provider, row.source), []).append(row)
    sources = [
        SourceRow(
            provider=provider,
            source=source,
            shap_sum=sum(item.shap for item in items if item.shap is not None),
            feature_count=len(items),
        )
        for (provider, source), items in grouped.items()
    ]
    sources.sort(key=lambda row: (-(row.shap_sum or 0.0), row.provider, row.source))
    return sources


def _psi_status(psi: Optional[float]) -> Optional[str]:
    if psi is None or pd.isna(psi):
        return None
    if psi > PSI_CRITICAL:
        return "risk"
    if psi >= PSI_WARNING:
        return "watch"
    return "good"


def _has_nonzero_shap_value(shap: Optional[float]) -> bool:
    return shap is not None and not pd.isna(shap) and shap != 0


def _overview_summary(
    *,
    model_rows: list[FeatureRow],
    found_rows: list[FeatureRow],
    sources: list[SourceRow],
    is_ensemble: Callable[[str], bool],
    generated_features: list[GeneratedFeatureMetadata],
    joined_ads_features_count: Optional[int],
    joined_ads_count: Optional[int],
) -> SearchResultsSummary:
    top_level = _unique_fold_model_inputs(generated_features)
    autofe_by_name = _autofe_generated_by_name(generated_features, is_ensemble)
    ads_ids = _contributed_ads_ids(generated_features, autofe_by_name)
    if top_level:
        counts = {"external": 0, "original": 0, "autofe": 0}
        for row in model_rows:
            counts[_model_row_kind(row)] += 1
        model_features = len(model_rows)
        external, original, autofe = counts["external"], counts["original"], counts["autofe"]
        contributed = len(ads_ids or set())
        psis = [row.psi for row in model_rows]
    else:
        model_features = len(model_rows) or None
        external = original = autofe = None
        contributed = (
            len(ads_ids)
            if ads_ids is not None
            else len({(row.provider, row.source) for row in found_rows if row.provider})
        )
        psis = [row.psi for row in model_rows]
    share, stable, psi_count = _stability_from_psi(psis)
    return SearchResultsSummary(
        relevant_features=len(found_rows),
        features_found=joined_ads_features_count if joined_ads_features_count is not None else len(found_rows),
        model_features=model_features,
        external_features=external,
        original_features=original,
        autofe_features=autofe,
        data_sources=len(sources),
        joined_sources=joined_ads_count if joined_ads_count is not None else len(sources),
        stable_features_share=share,
        contributed_sources=contributed,
        stable_features=stable,
        psi_features=psi_count,
    )


def _unique_fold_model_inputs(generated_features: list[GeneratedFeatureMetadata]) -> list[Column | Feature]:
    unique: dict[str, Column | Feature] = {}
    for meta in generated_features:
        feature = _parse_generated_feature(meta)
        if feature is None:
            continue
        for fold in _fold_model_nodes(feature):
            for child in fold.children:
                unique.setdefault(_top_level_key(child), child)
    return list(unique.values())


def _fold_model_nodes(feature: Feature) -> list[Feature]:
    if isinstance(feature.op, (CatboostModel, OnnxModel)):
        return [feature]
    nodes: list[Feature] = []
    for child in feature.children:
        if isinstance(child, Feature):
            nodes.extend(_fold_model_nodes(child))
    return nodes


def _top_level_key(node: Column | Feature) -> str:
    if isinstance(node, Column):
        return f"col:{node.name}"
    return f"fe:{node.to_formula()}"


def _autofe_generated_by_name(
    generated_features: list[GeneratedFeatureMetadata], is_ensemble: Callable[[str], bool]
) -> dict[str, GeneratedFeatureMetadata]:
    by_name: dict[str, GeneratedFeatureMetadata] = {}
    for meta in generated_features:
        feature = _parse_generated_feature(meta)
        if feature is None or isinstance(feature.op, (CatboostModel, EnsembleModel, OnnxModel)):
            continue
        name = feature.get_display_name(shorten=True, unhash=True, cache=False)
        if name and not is_ensemble(name):
            by_name.setdefault(name, meta)
    return by_name


def _parse_generated_feature(meta: GeneratedFeatureMetadata) -> Feature | None:
    try:
        parsed = Feature.from_formula(meta.formula)
        if not isinstance(parsed, Feature):
            return None
        orig_to_hashed = {column.original_name: column.hashed_name for column in meta.base_columns}
        return (
            parsed.set_display_index(meta.display_index or None)
            .set_alias(meta.alias)
            .set_op_params(meta.operator_params or {})
            .rename_columns(orig_to_hashed)
        )
    except Exception:
        return None


def _generated_feature_names(meta: GeneratedFeatureMetadata) -> set[str]:
    feature = _parse_generated_feature(meta)
    if feature is None:
        return set()
    return {feature.get_display_name(shorten=True, unhash=True, cache=False)}


def _model_row_kind(row: FeatureRow) -> str:
    source = (row.source or "").lower()
    if source.startswith("autofe") or source == GENERATED_SOURCE.lower():
        return "autofe"
    if source == CLIENT_SOURCE.lower():
        return "original"
    return "external"


def _contributed_ads_ids(
    generated_features: list[GeneratedFeatureMetadata],
    autofe_by_name: dict[str, GeneratedFeatureMetadata],
) -> Optional[set[str]]:
    ads_ids: Optional[set[str]] = None
    for meta in generated_features:
        feature = _parse_generated_feature(meta)
        if feature is None or not _fold_model_nodes(feature):
            continue
        if ads_ids is None:
            ads_ids = set()
        for column in meta.base_columns:
            ads_ids.update(_leaf_ads_ids(column, autofe_by_name))
    return ads_ids


def _leaf_ads_ids(column: BaseColumnMetadata, autofe_by_name: dict[str, GeneratedFeatureMetadata]) -> set[str]:
    autofe = autofe_by_name.get(column.hashed_name) or autofe_by_name.get(column.original_name)
    leaves = autofe.base_columns if autofe is not None else [column]
    return {leaf.ads_definition_id for leaf in leaves if leaf.ads_definition_id}


def _stability_from_psi(psis: list[Optional[float]]) -> tuple[Optional[float], Optional[int], Optional[int]]:
    values = [psi for psi in psis if psi is not None and not pd.isna(psi)]
    if not values:
        return None, None, None
    stable = sum(1 for psi in values if psi < _STABILITY_PSI)
    return stable / len(values), stable, len(values)
