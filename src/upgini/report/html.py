from __future__ import annotations

from html import escape
from typing import Optional

from upgini.report.data import QualitySample, ReportData, SampleStats, SearchResultsSummary

_PLACEHOLDER_SECTIONS = [
    "Performance",
    "Score analysis",
    "Score distribution",
    "Score stability (PSI)",
    "Model feature SHAP",
    "Search results",
    "Feature stability",
]


def generate_html_report(data: ReportData) -> str:
    meta = data.metadata
    download_name = f"upgini-report-{meta.search_id}.html"
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
      --header: #1c1917;
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
      background: var(--header);
      color: #fafaf9;
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .brand {{ display: flex; align-items: center; gap: 12px; min-width: 0; }}
    .logo {{
      width: 36px;
      height: 36px;
      border-radius: 8px;
      background: #292524;
      object-fit: contain;
      flex-shrink: 0;
    }}
    .logo-empty {{ border: 1px dashed #57534e; background: transparent; }}
    .chip {{
      background: #292524;
      color: #e7e5e4;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 13px;
    }}
    .download {{
      color: #fafaf9;
      text-decoration: none;
      border: 1px solid #a8a29e;
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 13px;
      white-space: nowrap;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 240px minmax(0, 1fr);
      gap: 16px;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px 16px 48px;
    }}
    aside {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px 18px;
      height: fit-content;
      color: var(--muted);
      font-size: 14px;
    }}
    aside p {{ margin: 0 0 10px; }}
    aside p:last-child {{ margin-bottom: 0; }}
    main {{ min-width: 0; }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 16px;
    }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    h1 {{ margin: 0 0 14px; font-size: 24px; font-weight: 600; }}
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
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
    .tabs button {{
      border: 1px solid var(--line);
      background: transparent;
      border-radius: 999px;
      padding: 4px 12px;
      cursor: pointer;
      color: var(--muted);
    }}
    .tabs button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ecfdf5;
    }}
    .chart-stubs {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 14px;
    }}
    .chart-stub {{
      border: 1px dashed var(--line);
      border-radius: 8px;
      min-height: 120px;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    @media (max-width: 800px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .chart-stubs {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      {_logo(meta.logo_url)}
      <span class="chip">search_id={escape(meta.search_id)}</span>
    </div>
    <a class="download" id="download-html" download="{escape(download_name)}">Download HTML</a>
  </header>
  <div class="layout">
    <aside>
      {_sidebar(data)}
    </aside>
    <main>
      {_overview_section(data)}
      {_quality_section(data.quality_by_sample)}
      {_feature_cards(data.summary)}
      {_sample_stats_section(data)}
      {''.join(_placeholder_section(title) for title in _PLACEHOLDER_SECTIONS)}
    </main>
  </div>
  <script>
    document.getElementById("download-html").href = URL.createObjectURL(
      new Blob(["<!DOCTYPE html>\\n" + document.documentElement.outerHTML], {{type: "text/html"}})
    );
    document.querySelectorAll("[data-tabs]").forEach(function (root) {{
      root.querySelectorAll("[data-tab]").forEach(function (btn) {{
        btn.addEventListener("click", function () {{
          var id = btn.getAttribute("data-tab");
          root.querySelectorAll("[data-tab]").forEach(function (other) {{
            other.classList.toggle("active", other === btn);
          }});
          root.querySelectorAll("[data-panel]").forEach(function (panel) {{
            panel.hidden = panel.getAttribute("data-panel") !== id;
          }});
        }});
      }});
    }});
  </script>
</body>
</html>
"""


def _logo(logo_url: Optional[str]) -> str:
    if logo_url:
        return f'<img class="logo" src="{escape(logo_url)}" alt="" />'
    return '<div class="logo logo-empty" aria-hidden="true"></div>'


def _sidebar(data: ReportData) -> str:
    meta = data.metadata
    duration = meta.search_duration or "—"
    sources = _dash(data.summary.data_sources)
    rows = f"{meta.reference_rows:,}" if meta.reference_rows is not None else "—"
    return (
        f"<p>Search completed · {escape(duration)}</p>"
        f"<p>· {escape(sources)} sources</p>"
        f"<p>{escape(rows)} rows processed</p>"
    )


def _overview_section(data: ReportData) -> str:
    meta = data.metadata
    keys = ", ".join(meta.search_keys) if meta.search_keys else "—"
    samples = ", ".join(meta.samples) if meta.samples else "—"
    return f"""<section>
      <h1>FIT completed successfully</h1>
      <div class="meta-grid">
        {_meta_item("Generated", meta.generated_at)}
        {_meta_item("PyLib version", meta.pylib_version)}
        {_meta_item("Keys", keys)}
        {_meta_item("Samples", samples)}
      </div>
    </section>"""


def _feature_cards(summary: SearchResultsSummary) -> str:
    stable = (
        f"{summary.stable_features_share * 100:.0f}%"
        if summary.stable_features_share is not None
        else "—"
    )
    return f"""<section>
      <div class="cards">
        {_card("Features found", _dash(summary.relevant_features))}
        {_card("Used in model", _dash(summary.model_features))}
        {_card("Data sources", _dash(summary.data_sources))}
        {_card("Stability", stable)}
      </div>
    </section>"""


def _quality_section(samples: list[QualitySample]) -> str:
    if not samples:
        return '<section><h2>Key result</h2><p class="empty">Metrics were not calculated for this search.</p></section>'
    metrics = list(dict.fromkeys(sample.metric or "score" for sample in samples))
    tabs = []
    panels = []
    for index, metric in enumerate(metrics):
        active = " active" if index == 0 else ""
        hidden = "" if index == 0 else " hidden"
        tabs.append(
            f'<button type="button" class="{active.strip()}" data-tab="{escape(metric)}">{escape(metric)}</button>'
        )
        rows = []
        for sample in samples:
            if (sample.metric or "score") != metric:
                continue
            rows.append(
                "<tr>"
                f"<td>{escape(sample.evaluation_scope)}</td>"
                f"<td class='num'>{escape(_fmt_score(sample.baseline))}</td>"
                f"<td class='num'>{escape(_fmt_score(sample.enriched))}</td>"
                f"<td class='num'>{escape(_fmt_score(sample.std))}</td>"
                f"<td class='num uplift'>{escape(_dash(sample.uplift))}</td>"
                f"<td class='num uplift'>{escape(_dash(sample.relative_uplift))}</td>"
                "</tr>"
            )
        panels.append(
            f'<div data-panel="{escape(metric)}"{hidden}>'
            "<table>"
            "<thead><tr>"
            "<th>Scope</th><th class='num'>Baseline</th><th class='num'>Enriched</th>"
            "<th class='num'>Std</th><th class='num'>Uplift, abs</th><th class='num'>Uplift, %</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table></div>"
        )
    return f"""<section data-tabs>
      <h2>Key result</h2>
      <div class="tabs">{''.join(tabs)}</div>
      {''.join(panels)}
    </section>"""


def _sample_stats_section(data: ReportData) -> str:
    stats = data.sample_stats or [SampleStats(sample=name) for name in data.metadata.samples] or [
        SampleStats(sample="Train")
    ]
    total = (
        f"{data.metadata.reference_rows:,} total rows"
        if data.metadata.reference_rows is not None
        else "— total rows"
    )
    tabs = []
    panels = []
    for index, sample in enumerate(stats):
        sample_id = str(index)
        active = " active" if index == 0 else ""
        hidden = "" if index == 0 else " hidden"
        tabs.append(
            f'<button type="button" class="{active.strip()}" data-tab="{sample_id}">{escape(sample.sample)}</button>'
        )
        items = [
            _meta_item("Count", _format_int(sample.rows)),
            _meta_item("Date range", _dash(sample.date_range)),
            _meta_item("Labeled", _format_int(sample.labeled)),
            _meta_item("Unlabeled", _format_int(sample.unlabeled)),
            _meta_item("Mean target", _dash(sample.mean_target)),
        ]
        if data.is_binary:
            items.insert(4, _meta_item("Positive", _format_int(sample.positive)))
        panels.append(
            f'<div data-panel="{sample_id}"{hidden}>'
            f'<div class="meta-grid">{"".join(items)}</div>'
            '<div class="chart-stubs">'
            '<div class="chart-stub">Monthly rows</div>'
            '<div class="chart-stub">Monthly mean target</div>'
            "</div></div>"
        )
    return f"""<section data-tabs>
      <h2>Sample stats</h2>
      <div class="chip" style="margin-bottom:12px">{escape(total)}</div>
      <div class="tabs">{''.join(tabs)}</div>
      {''.join(panels)}
    </section>"""


def _placeholder_section(title: str) -> str:
    return f"<section><h2>{escape(title)}</h2></section>"


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


def _fmt_score(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def _format_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "—"
