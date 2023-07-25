from typing import Callable
import pandas as pd
import urllib.parse


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


def display_html_dataframe(df: pd.DataFrame, internal_df: pd.DataFrame, header: str):
    if not ipython_available():
        print(header)
        print(internal_df)
        return

    from IPython.display import HTML, display

    def map_to_td(value) -> str:
        if isinstance(value, float):
            return f"<td class='upgini-number'>{value:.4f}</td>"
        else:
            return f"<td class='upgini-text'>{value}</td>"

    table_tsv = urllib.parse.quote(internal_df.to_csv(index=False, sep="\t"), safe=",")

    table_html = (
        "<table class='upgini-df'>"
        + "<thead>"
        + "".join(f"<th>{col}</th>" for col in df.columns)
        + "</thead>"
        + "<tbody>"
        + "".join("<tr>" + "".join(map(map_to_td, row[1:])) + "</tr>" for row in df.itertuples())
        + "</tbody>"
        + "</table>"
    )

    result_html = (
        """<style>
            .upgini-df thead th {
                font-weight:bold;
                text-align: right;
                padding: 0.5em;
            }

            .upgini-df td {
                padding: 0.5em;
            }

            .upgini-text {
                text-align: right;
            }

            .upgini-number {
                text-align: center;
            }
        </style>
        """ +
        f"""
        <h2>{header}</h2>
        <div style="text-align: right">
            <button onclick=navigator.clipboard.writeText(decodeURI('{table_tsv}'))>Copy</button>
            <a href='mailto:<put email for share>?subject={header}&body=<Paste search result here>'>
                <button>Share</button>
            </a>
        </div>
        """ + table_html
    )
    display(HTML(result_html))


# def show_button_download_pdf(source: str, title="Download as PDF"):
#     from xhtml2pdf import pisa

#     file_name = f"report-{uuid.uuid4()}.pdf"
#     with open(file_name, "wb") as output:
#         pisa.CreatePDF(
#             src=source,
#             dest=output
#         )

#     import base64
#     from IPython.display import HTML

#     with open(file_name, "rb") as f:
#         b64 = base64.b64encode(f.read())
#         payload = b64.decode()
#         html = '<a download="{filename}" href="data:application/pdf;base64,{payload}" target="_blank">{title}</a>'
#         html = html.format(payload=payload, title=title, filename=file_name)
#         return HTML(html)
