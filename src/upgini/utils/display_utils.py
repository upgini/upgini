import base64
import math
import textwrap
import urllib.parse
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Callable, List, Optional

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
                border-bottom: 2px solid black;
            }}

            .upgini-df tbody td {{
                padding: 0.5em;
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
            <div style="text-align: right">
                <button onclick=navigator.clipboard.writeText(decodeURI('{table_tsv}'))>\U0001F4C2 Copy</button>
                <a href='mailto:<Share with...>?subject={email_subject}&body={table_tsv}'>
                    <button>\U0001F4E8 Share</button>
                </a>
            </div>
            {table_html}
        </div>
        """
    display(HTML(result_html))


def make_html_report(
    relevant_features_df: pd.DataFrame,
    relevant_datasources_df: pd.DataFrame,
    metrics_df: Optional[pd.DataFrame],
    autofe_descriptions_df: Optional[pd.DataFrame],
    search_id: str,
    email: Optional[str] = None,
    search_keys: Optional[List[str]] = None,
):
    # relevant_features_df = relevant_features_df.copy()
    # relevant_features_df["Feature name"] = relevant_features_df["Feature name"].apply(
    #     lambda x: "*" + x if x.contains("_autofe_") else x
    # )
    relevant_datasources_df = relevant_datasources_df.copy()
    relevant_datasources_df["action"] = (
        f"""<a href="https://upgini.com/requet-a-quote?search-id={search_id}">"""
        """<button type="button">Request a quote</button></a>"""
    )
    relevant_datasources_df.rename(columns={"action": "&nbsp;"}, inplace=True)
    return f"""<html>
        <head>
            <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
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

                @font-face {{
                    font-family: "Alice-Regular";
                    src: url("/fonts/Alice-Regular.ttf") format("truetype");
                }}

                body {{
                    font-family: "Alice-Regular", Arial, sans-serif;
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
                © Upgini, Dubai, UAE</br>
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
            {
                f"The following primary keys was used for data search: {search_keys}"
                if search_keys is not None
                else ""
            }
            {"<h3>All relevant features. Accuracy after enrichment</h3>" + make_table(metrics_df)
             if metrics_df is not None
             else ""
            }
            <h3>Relevant data sources</h3>
            {make_table(relevant_datasources_df)}
            <h3>All relevant features. Listing</h3>
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
):
    if not ipython_available():
        return

    report = make_html_report(
        relevant_features_df, relevant_datasources_df, metrics_df, autofe_descriptions_df, search_id, email, search_keys
    )

    if len(relevant_features_df) > 0:
        show_button_download_pdf(report)


def show_button_download_pdf(source: str, title="\U0001F4CA Download PDF report"):
    from IPython.display import HTML, display

    file_name = f"upgini-report-{uuid.uuid4()}.pdf"
    with open(file_name, "wb") as output:
        pisa.CreatePDF(src=BytesIO(source.encode("UTF-8")), dest=output)

    with open(file_name, "rb") as f:
        b64 = base64.b64encode(f.read())
        payload = b64.decode()
        html = f"""<a download="{file_name}" href="data:application/pdf;base64,{payload}" target="_blank">
        <button>{title}</button></a>"""
        display(HTML(html))


def show_request_quote_button():
    if not ipython_available():
        print("https://upgini.com/requet-a-quote")
    else:
        import ipywidgets as widgets
        from IPython.display import Javascript, display

        button = widgets.Button(description="Request a quote", button_style="danger")

        def on_button_clicked(b):
            display(Javascript('window.open("https://upgini.com/requet-a-quote");'))

        button.on_click(on_button_clicked)

        display(button)
