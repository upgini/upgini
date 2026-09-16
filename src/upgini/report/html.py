from __future__ import annotations

from html import escape
from typing import Optional

from upgini.report.data import FeatureRow, QualitySample, ReportData, SearchResultsSummary, SourceRow


def generate_html_report(data: ReportData) -> str:
    meta = data.metadata
    keys = ", ".join(meta.search_keys) if meta.search_keys else "—"
    samples = ", ".join(meta.samples) if meta.samples else "—"
    duration = meta.search_duration or "—"
    rows = f"{meta.reference_rows:,}" if meta.reference_rows is not None else "—"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Upgini search report {escape(meta.search_id)}</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --ink: #1c1917;
      --muted: #57534e;
      --card: #fffdf8;
      --line: #e7e0d4;
      --accent: #0f766e;
      --accent-ink: #134e4a;
      --chip: #ecfdf5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: #1c1917;
      color: #fafaf9;
      padding: 28px 32px 24px;
    }}
    header .kicker {{
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-size: 12px;
      color: #a8a29e;
      margin-bottom: 8px;
    }}
    h1 {{ margin: 0 0 12px; font-size: 28px; font-weight: 600; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{
      background: #292524;
      color: #e7e5e4;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 13px;
    }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 24px 20px 48px; }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 16px;
    }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .meta-grid, .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .meta-item .label, .card .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .meta-item .value, .card .value {{
      font-size: 20px;
      font-weight: 600;
      margin-top: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .uplift {{ color: var(--accent-ink); font-weight: 600; }}
    .empty {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <div class="kicker">Upgini search report</div>
    <h1>FIT completed successfully</h1>
    <div class="chips">
      <span class="chip">search_id={escape(meta.search_id)}</span>
      <span class="chip">{escape(keys)}</span>
    </div>
  </header>
  <main>
    <section>
      <h2>Overview</h2>
      <div class="meta-grid">
        {_meta_item("Generated", meta.generated_at)}
        {_meta_item("PyLib version", meta.pylib_version)}
        {_meta_item("Search duration", duration)}
        {_meta_item("Rows processed", rows)}
        {_meta_item("Samples", samples)}
      </div>
    </section>
    {_summary_section(data.summary)}
    {_quality_section(data.quality_by_sample)}
    {_features_section(data.features)}
    {_sources_section(data.sources)}
    {_autofe_section(data.autofe)}
  </main>
</body>
</html>
"""


def _meta_item(label: str, value: str) -> str:
    return (
        f'<div class="meta-item"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(str(value))}</div></div>'
    )


def _card(label: str, value) -> str:
    return (
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(str(value))}</div></div>'
    )


def _dash(value: Optional[object]) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def _summary_section(summary: SearchResultsSummary) -> str:
    if all(
        value is None
        for value in (
            summary.relevant_features,
            summary.model_features,
            summary.data_sources,
            summary.stable_features_share,
        )
    ):
        return ""
    stable = (
        f"{summary.stable_features_share * 100:.0f}%"
        if summary.stable_features_share is not None
        else "—"
    )
    return f"""<section>
      <h2>Summary</h2>
      <div class="cards">
        {_card("Features found", _dash(summary.relevant_features))}
        {_card("Used in model", _dash(summary.model_features))}
        {_card("Data sources", _dash(summary.data_sources))}
        {_card("Stable features", stable)}
      </div>
    </section>"""


def _quality_section(samples: list[QualitySample]) -> str:
    if not samples:
        return '<section><h2>Key result</h2><p class="empty">Metrics were not calculated for this search.</p></section>'
    metric = samples[0].metric or "score"
    rows = []
    for sample in samples:
        rows.append(
            "<tr>"
            f"<td>{escape(sample.evaluation_scope)}</td>"
            f"<td class='num'>{escape(_dash(sample.baseline))}</td>"
            f"<td class='num'>{escape(_dash(sample.enriched))}</td>"
            f"<td class='num uplift'>{escape(_dash(sample.uplift))}</td>"
            f"<td class='num uplift'>{escape(_dash(sample.relative_uplift))}</td>"
            "</tr>"
        )
    return f"""<section>
      <h2>Key result · {escape(metric)}</h2>
      <table>
        <thead>
          <tr>
            <th>Scope</th>
            <th class="num">Baseline</th><th class="num">Enriched</th>
            <th class="num">Uplift, abs</th><th class="num">Uplift, %</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _features_section(features: list[FeatureRow]) -> str:
    if not features:
        return ""
    rows = []
    for idx, feature in enumerate(features, start=1):
        rows.append(
            "<tr>"
            f"<td class='num'>{idx}</td>"
            f"<td>{escape(feature.name)}</td>"
            f"<td>{escape(feature.provider)}</td>"
            f"<td>{escape(feature.source)}</td>"
            f"<td class='num'>{escape(_dash(feature.shap))}</td>"
            f"<td class='num'>{escape(_dash(feature.psi))}</td>"
            f"<td class='num'>{escape(_dash(feature.drift))}</td>"
            f"<td class='num'>{escape(_dash(feature.coverage))}</td>"
            "</tr>"
        )
    return f"""<section>
      <h2>Search results · features</h2>
      <table>
        <thead>
          <tr>
            <th class="num">#</th><th>Feature</th><th>Provider</th><th>Source</th>
            <th class="num">SHAP</th><th class="num">PSI</th>
            <th class="num">Drift</th><th class="num">Coverage %</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _sources_section(sources: list[SourceRow]) -> str:
    if not sources:
        return ""
    rows = []
    for source in sources:
        rows.append(
            "<tr>"
            f"<td>{escape(source.provider)}</td>"
            f"<td>{escape(source.source)}</td>"
            f"<td class='num'>{escape(_dash(source.shap_sum))}</td>"
            f"<td class='num'>{escape(_dash(source.feature_count))}</td>"
            "</tr>"
        )
    return f"""<section>
      <h2>Search results · sources</h2>
      <table>
        <thead>
          <tr><th>Provider</th><th>Source</th><th class="num">SHAP sum</th><th class="num">Features</th></tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _autofe_section(autofe: list[dict[str, str]]) -> str:
    if not autofe:
        return ""
    columns = list(autofe[0].keys())
    head = "".join(f"<th>{escape(col)}</th>" for col in columns)
    rows = []
    for item in autofe:
        rows.append("<tr>" + "".join(f"<td>{escape(item.get(col, ''))}</td>" for col in columns) + "</tr>")
    return f"""<section>
      <h2>AutoFE</h2>
      <table>
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""
