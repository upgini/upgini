import base64
import urllib.parse
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional
import textwrap

import pandas as pd
from xhtml2pdf import pisa


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
            return f"<td class='upgini-number'>{value:.4f}</td>"
        elif isinstance(value, int):
            return f"<td class='upgini-number'>{value}</td>"
        else:
            if wrap_long_string is not None and len(value) > wrap_long_string:
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


def display_html_dataframe(df: pd.DataFrame, internal_df: pd.DataFrame, header: str):
    if not ipython_available():
        print(header)
        print(internal_df)
        return

    from IPython.display import HTML, display

    table_tsv = urllib.parse.quote(internal_df.to_csv(index=False, sep="\t"), safe=",")
    email_subject = "Relevant external data sources from Upgini.com"

    table_html = make_table(df)

    result_html = f"""<style>
            .upgini-df thead th {{
                font-weight:bold;
                text-align: center;
                padding: 0.5em;
            }}

            .upgini-df tbody td {{
                padding: 0.5em;
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
            <div style="text-align: right">
                <button onclick=navigator.clipboard.writeText(decodeURI('{table_tsv}'))>Copy</button>
                <a href='mailto:<Share with...>?subject={email_subject}&body={table_tsv}'>
                    <button>Share</button>
                </a>
            </div>
            {table_html}
        </div>
        """
    display(HTML(result_html))


def prepare_and_show_report(
    relevant_features_df: pd.DataFrame,
    relevant_datasources_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    search_id: str,
    email: Optional[str],
):
    if not ipython_available():
        return

    report = f"""<html>
        <head>
            <style>
                @page {{
                    size: a4 portrait;
                    @frame header_frame {{
                        -pdf-frame-content: header_content;
                        left: 10pt; width: 574pt; top: 10pt; height: 40pt;
                        /*-pdf-frame-border: 1;*/
                    }}
                    @frame content_frame {{
                        left: 10pt; width: 574pt; top: 50pt; height: 752pt;
                        /*-pdf-frame-border: 1;*/
                    }}
                    @frame footer_frame {{
                        -pdf-frame-content: footer_content;
                        left: 10pt; width: 574pt; top: 802pt; height: 30pt;
                        /*-pdf-frame-border: 1;*/
                    }}
                }}

                #header_content {{
                    background-color: black;
                    color: white;
                    text-align: center;
                    padding-top: 6pt;
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
                © Upgini, DWTC, Dubai, UAE</br>
                sales@upgini.com
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
            <h3>All relevant features. Accuracy after enrichment</h3>
            {make_table(metrics_df)}
            <h3>All relevant features. Listing</h3>
            {make_table(relevant_features_df, wrap_long_string=25)}
            <h3>Relevant data sources</h3>
            {make_table(relevant_datasources_df)}
            <p>To buy found data sources, please contact: <a href='mailto:sales@upgini.com'>sales@upgini.com</a></p>
            <p>Best regards, </br><b>Upgini Team</b></p>
        </body>
    </html>"""

    show_button_download_pdf(report)


def show_button_download_pdf(source: str, title="Download PDF report"):
    from IPython.display import HTML, display

    file_name = f"report-{uuid.uuid4()}.pdf"
    with open(file_name + ".html", "w") as f:
        f.write(source)
    with open(file_name, "wb") as output:
        pisa.CreatePDF(src=source, dest=output)

    with open(file_name, "rb") as f:
        b64 = base64.b64encode(f.read())
        payload = b64.decode()
        html = f"""<a download="{file_name}" href="data:application/pdf;base64,{payload}" target="_blank">
        <button>{title}</button></a>"""
        display(HTML(html))
