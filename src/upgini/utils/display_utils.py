from typing import Callable
import pandas as pd


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


def display_html_dataframe(df: pd.DataFrame):
    from IPython.display import HTML, display

    def map_to_td(value) -> str:
        if isinstance(value, float):
            return f"<td class='upgini-number'>{value:.6f}</td>"
        else:
            return f"<td class='upgini-text'>{value}</td>"

    table_str = (
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
        </style>"""
        + "<table class='upgini-df'>"
        + "<thead>"
        + "".join(f"<th>{col}</th>" for col in df.columns)
        + "</thead>"
        + "<tbody>"
        + "".join("<tr>" + "".join(map(map_to_td, row[1:])) + "</tr>" for row in df.itertuples())
        + "</tbody>"
        + "</table>"
    )
    display(HTML(table_str))
