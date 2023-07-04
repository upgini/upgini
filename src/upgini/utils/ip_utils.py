import logging
from typing import Dict, List, Optional

import pandas as pd
from requests import get

from upgini.metadata import SearchKey
# from upgini.resource_bundle import bundle
# from upgini.utils.track_info import get_track_metrics


class IpToCountrySearchKeyConverter:

    url = "http://ip-api.com/json/{}"

    def __init__(
        self,
        search_keys: Dict[str, SearchKey],
        logger: Optional[logging.Logger] = None,
    ):
        self.search_keys = search_keys
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")
        self.generated_features: List[str] = []

    def _get_country_code(self, ip: str) -> str:
        try:
            response = get(self.url.format(ip)).json()
            return response["countryCode"]
        except Exception:
            return None

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO temporary turn off country detection

        # track_metrics = get_track_metrics()
        # if (
        #     track_metrics is not None
        #     and "ip" in track_metrics
        #     and track_metrics["ip"] is not None
        #     and track_metrics["ip"] != "0.0.0.0"
        # ):
        #     country_code = self._get_country_code(track_metrics["ip"])  # TODO check that country_code is not None!
        #     msg = bundle.get("country_auto_determined").format(country_code)
        #     print(msg)
        #     self.logger.info(msg)
        #     df["country_code"] = country_code
        #     self.search_keys["country_code"] = SearchKey.COUNTRY

        return df
