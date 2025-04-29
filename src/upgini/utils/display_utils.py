import base64
import math
import textwrap
import urllib.parse
import uuid
from datetime import datetime, timezone
from io import StringIO
from typing import Callable, List, Optional

import pandas as pd
from xhtml2pdf import pisa

from upgini.__about__ import __version__


def ipython_available() -> bool:
    try:
        _ = get_ipython()  # type: ignore
        return True
    except NameError:
        return False


def do_without_pandas_limits(func: Callable):
    prev_max_rows = pd.options.display.max_rows
    prev_max_columns = pd.options.display.max_columns
    prev_max_colwidth = pd.options.display.max_colwidth
    prev_width = pd.options.display.width

    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None
    pd.options.display.width = 150

    try:
        func()
    finally:
        pd.options.display.max_rows = prev_max_rows
        pd.options.display.max_columns = prev_max_columns
        pd.options.display.max_colwidth = prev_max_colwidth
        pd.options.display.width = prev_width


def make_table(df: pd.DataFrame, wrap_long_string=None) -> str:
    def map_to_td(value) -> str:
        if isinstance(value, float):
            if value is None or not math.isfinite(value):
                value = "&nbsp;"
            else:
                value = f"{value:.4f}"
            return f"<td class='upgini-number'>{value}</td>"
        elif isinstance(value, int):
            if value is None:
                value = "&nbsp;"
            return f"<td class='upgini-number'>{value}</td>"
        else:
            if value is None or len(value) == 0 or value == "nan":
                value = "&nbsp;"
            elif wrap_long_string is not None and len(value) > wrap_long_string and " " not in value:
                value = "</br>".join(textwrap.wrap(value, wrap_long_string))

            return f"<td class='upgini-text'>{value}</td>"

    return (
        "<table class='upgini-df'>"
        + "<thead>"
        + "".join(f"<th>{col}</th>" for col in df.columns)
        + "</thead>"
        + "<tbody>"
        + "".join("<tr>" + "".join(map(map_to_td, row[1:])) + "</tr>" for row in df.itertuples())
        + "</tbody>"
        + "</table>"
    )


def display_html_dataframe(
    df: pd.DataFrame, internal_df: pd.DataFrame, header: str, display_id: Optional[str] = None, display_handle=None
):
    if not ipython_available():
        print(header)
        print(internal_df)
        return

    from IPython.display import HTML, display

    try:
        table_tsv = urllib.parse.quote(internal_df.to_csv(index=False, sep="\t"), safe=",")
    except Exception:
        table_tsv = None
    email_subject = "Relevant external data sources from Upgini.com"

    if table_tsv is not None:
        copy_and_share = f"""
            <div style="text-align: right">
                <button onclick=navigator.clipboard.writeText(decodeURI('{table_tsv}'))>\U0001f4c2 Copy</button>
                <a href='mailto:<Share with...>?subject={email_subject}&body={table_tsv}'>
                    <button>\U0001f4e8 Share</button>
                </a>
            </div>"""
    else:
        copy_and_share = ""

    table_html = make_table(df)

    result_html = f"""<style>
            .upgini-df thead th {{
                font-weight:bold;
                text-align: center;
                padding: 0.5em;
                border-bottom: 2px solid black;
            }}

            .upgini-df tbody td {{
                padding: 0.5em;
                color: black;
            }}

            .upgini-df tbody tr:nth-child(odd) {{
                background-color: #ffffff;
            }}

            .upgini-df tbody tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}

            .upgini-text {{
                text-align: left;
            }}

            .upgini-number {{
                text-align: center;
            }}
        </style>
        <h2>{header}</h2>
        <div style="display:flex; flex-direction:column; align-items:flex-end; width: fit-content;">
            {copy_and_share}
            {table_html}
        </div>
        """
    if display_handle:
        return display_handle.update(HTML(result_html))
    else:
        return display(HTML(result_html), display_id=display_id)


def make_html_report(
    relevant_features_df: pd.DataFrame,
    relevant_datasources_df: pd.DataFrame,
    metrics_df: Optional[pd.DataFrame],
    autofe_descriptions_df: Optional[pd.DataFrame],
    search_id: str,
    email: Optional[str] = None,
    search_keys: Optional[List[str]] = None,
) -> str:
    # relevant_features_df = relevant_features_df.copy()
    # relevant_features_df["Feature name"] = relevant_features_df["Feature name"].apply(
    #     lambda x: "*" + x if x.contains("_autofe_") else x
    # )
    relevant_datasources_df = relevant_datasources_df.copy()
    relevant_datasources_df["action"] = (
        f"""<a href="https://upgini.com/request-a-quote?search-id={search_id}">"""
        """<button type="button">Request a quote</button></a>"""
    )
    relevant_datasources_df.rename(columns={"action": "&nbsp;"}, inplace=True)

    try:
        from importlib.resources import files

        font_path = files("upgini.utils").joinpath("Roboto-Regular.ttf")
    except Exception:
        from pkg_resources import resource_filename

        font_path = resource_filename("upgini.utils", "Roboto-Regular.ttf")

    return f"""<html>
        <head>
            <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
            <meta charset="UTF-8">
            <style>
                @page {{
                    size: a4 portrait;
                    @frame header_frame {{
                        -pdf-frame-content: header_content;
                        left: 10pt; width: 574pt; top: 10pt; height: 40pt;
                        /*-pdf-frame-border: 1;*/
                    }}
                    @frame content_frame {{
                        left: 10pt; width: 574pt; top: 50pt; height: 742pt;
                        /*-pdf-frame-border: 1;*/
                    }}
                    @frame footer_frame {{
                        -pdf-frame-content: footer_content;
                        left: 10pt; width: 574pt; top: 802pt; height: 40pt;
                        /*-pdf-frame-border: 1;*/
                    }}
                }}

                @font-face {{
                    font-family: "Roboto";
                    src: url("{font_path}") format("truetype");
                }}

                body {{
                    font-family: "Roboto", sans-serif;
                    font-weight: 400;
                    font-style: normal;
                }}

                #header_content {{
                    background-color: black;
                    color: white;
                    text-align: center;
                    padding-top: 7pt;
                    font-size: 15pt;
                    height: 38pt;
                }}

                h1 {{
                    text-align: center;
                }}

                .upgini-df {{
                    font-size: 5pt;
                    border: 1px solid black;
                    table-layout: auto !important;
                }}

                .upgini-df thead th {{
                    font-weight:bold;
                    text-align: center;
                    padding: 0.5em;
                    width: auto !important;
                    border: 1px solid black;
                    border-collapse: collapse;
                }}

                .upgini-df tbody td {{
                    padding: 0.5em;
                    width: auto !important;
                    border: 1px solid black;
                    border-collapse: collapse;
                    -pdf-keep-in-frame-mode: shrink;
                }}

                .upgini-text {{
                    text-align: left;
                }}

                .upgini-number {{
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div id="header_content">UPGINI</div>
            <div id="footer_content">
                © Upgini</br>
                sales@upgini.com</br>
                Launched by version {__version__}
            </div>

            <h1>Data search report</h1>
            <p style="text-align: right">
                <b>Datetime:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</br>
                <b>Search ID:</b> {search_id}</br>
                {"<b>Initiator:</b> " + email if email else ""}
            </p>
            <p>This report was generated automatically by Upgini.</p>
            <p>The report shows a listing of relevant features for your
            ML task and accuracy metrics after enrichment.</p>
            {
                f"The following primary keys was used for data search: {search_keys}"
                if search_keys is not None
                else ""
            }
            {"<h3>All relevant features. Accuracy after enrichment</h3>" + make_table(metrics_df)
             if metrics_df is not None
             else ""
            }
            {"<h3>Relevant data sources</h3>" + make_table(relevant_datasources_df)
             if len(relevant_datasources_df) > 0
             else ""
            }
            <h3>All relevant features. Listing ({len(relevant_features_df)} items)</h3>
            {make_table(relevant_features_df, wrap_long_string=25)}
            {"<h3>Description of AutoFE feature names</h3>" + make_table(autofe_descriptions_df, wrap_long_string=25)
             if autofe_descriptions_df is not None
             else ""
            }
            <p>To buy found data sources, please contact: <a href='mailto:sales@upgini.com'>sales@upgini.com</a></p>
            <p>Best regards, </br><b>Upgini Team</b></p>
        </body>
    </html>"""


def prepare_and_show_report(
    relevant_features_df: pd.DataFrame,
    relevant_datasources_df: pd.DataFrame,
    metrics_df: Optional[pd.DataFrame],
    autofe_descriptions_df: Optional[pd.DataFrame],
    search_id: str,
    email: Optional[str],
    search_keys: Optional[List[str]] = None,
    display_id: Optional[str] = None,
    display_handle=None,
):
    if not ipython_available():
        return

    report = make_html_report(
        relevant_features_df, relevant_datasources_df, metrics_df, autofe_descriptions_df, search_id, email, search_keys
    )

    if len(relevant_features_df) > 0:
        return show_button_download_pdf(report, display_id=display_id, display_handle=display_handle)


def show_button_download_pdf(
    source: str, title="\U0001f4ca Download PDF report", display_id: Optional[str] = None, display_handle=None
):
    from IPython.display import HTML, display

    file_name = f"upgini-report-{uuid.uuid4()}.pdf"

    # from weasyprint import HTML

    # html = HTML(string=source)
    # html.write_pdf(file_name)
    with open(file_name, "wb") as output:
        pisa.CreatePDF(src=StringIO(source), dest=output, encoding="UTF-8")

    with open(file_name, "rb") as f:
        b64 = base64.b64encode(f.read())
        payload = b64.decode()
        html = f"""<a download="{file_name}" href="data:application/pdf;base64,{payload}" target="_blank">
        <button>{title}</button></a>"""
        if display_handle is not None:
            display_handle.update(HTML(html))
        else:
            return display(HTML(html), display_id=display_id)


def show_request_quote_button():
    if not ipython_available():
        print("https://upgini.com/request-a-quote")
    else:
        import ipywidgets as widgets
        from IPython.display import Javascript, display

        button = widgets.Button(description="Request a quote", button_style="danger")

        def on_button_clicked(b):
            display(Javascript('window.open("https://upgini.com/request-a-quote");'))

        button.on_click(on_button_clicked)

        display(button)
