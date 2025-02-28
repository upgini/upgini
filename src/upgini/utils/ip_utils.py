import logging
from ipaddress import IPv4Address, IPv6Address, _BaseAddress, ip_address
from typing import Dict, List, Optional, Union

import pandas as pd
from requests import get

from upgini.errors import ValidationError
from upgini.metadata import SearchKey
from upgini.resource_bundle import ResourceBundle, get_custom_bundle

# from upgini.resource_bundle import bundle
# from upgini.utils.track_info import get_track_metrics


class IpSearchKeyConverter:
    def __init__(
        self,
        ip_column: str,
        search_keys: Dict[str, SearchKey],
        columns_renaming: Dict[str, str],
        unnest_search_keys: Optional[List[str]] = None,
        bundle: Optional[ResourceBundle] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.ip_column = ip_column
        self.search_keys = search_keys
        self.columns_renaming = columns_renaming
        self.unnest_search_keys = unnest_search_keys
        self.bundle = bundle or get_custom_bundle()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")

    @staticmethod
    def _ip_to_int(ip: Optional[_BaseAddress]) -> Optional[int]:
        if ip is None:
            return None
        try:
            if isinstance(ip, (IPv4Address, IPv6Address)):
                return int(ip)
        except Exception:
            pass

    @staticmethod
    def _ip_to_binary(ip: Optional[_BaseAddress]) -> Optional[bytes]:
        if ip is None:
            return None
        try:
            if isinstance(ip, IPv6Address) and ip.ipv4_mapped is not None:
                return ip.ipv4_mapped.packed
            else:
                return ip.packed
        except Exception:
            pass

    @staticmethod
    def _ip_to_prefix(ip: Optional[_BaseAddress]) -> Optional[str]:
        if ip is None:
            return None
        try:
            if isinstance(ip, IPv6Address):
                if ip.ipv4_mapped is not None:
                    return ".".join(ip.ipv4_mapped.exploded.split(".")[:2])
                return ":".join(ip.exploded.split(":")[:2])  # TODO use 3 in future
            else:
                return ".".join(ip.exploded.split(".")[:2])
        except Exception:
            pass

    @staticmethod
    def _ip_to_int_str(ip: Optional[_BaseAddress]) -> Optional[str]:
        try:
            if isinstance(ip, (IPv4Address, IPv6Address)):
                return str(int(ip))
        except Exception:
            pass

    @staticmethod
    def _safe_ip_parse(ip: Union[str, int, IPv4Address, IPv6Address]) -> Optional[_BaseAddress]:
        try:
            return ip_address(ip)
        except ValueError:
            pass

    # @staticmethod
    # def _is_ipv4(ip: Optional[_BaseAddress]):
    #     return ip is not None and (
    #         isinstance(ip, IPv4Address) or (isinstance(ip, IPv6Address) and ip.ipv4_mapped is not None)
    #     )

    # @staticmethod
    # def _to_ipv4(ip: Optional[_BaseAddress]) -> Optional[IPv4Address]:
    #     if isinstance(ip, IPv4Address):
    #         return ip
    #     return None

    @staticmethod
    def _to_ipv6(ip: Optional[_BaseAddress]) -> Optional[IPv6Address]:
        if isinstance(ip, IPv6Address):
            return ip
        if isinstance(ip, IPv4Address):
            return IPv6Address("::ffff:" + str(ip))
        return None

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert ip address to int"""
        self.logger.info("Convert ip address to int")
        original_ip = self.columns_renaming[self.ip_column]

        df[self.ip_column] = df[self.ip_column].apply(self._safe_ip_parse)
        if df[self.ip_column].isnull().all():
            raise ValidationError(self.bundle.get("invalid_ip").format(self.ip_column))

        # legacy support
        # ipv4 = self.ip_column + "_v4"
        # df[ipv4] = df[self.ip_column].apply(self._to_ipv4).apply(self._ip_to_int).astype("Int64")
        # self.search_keys[ipv4] = SearchKey.IP
        # self.columns_renaming[ipv4] = original_ip

        # ipv6 = self.ip_column + "_v6"
        # df[ipv6] = (
        #     df[self.ip_column]
        #     .apply(self._to_ipv6)
        #     .apply(self._ip_to_int_str)
        #     .astype("string")
        #     # .str.replace(".0", "", regex=False)
        # )
        ip_binary = self.ip_column + "_binary"
        df[ip_binary] = df[self.ip_column].apply(self._ip_to_binary)
        ip_prefix_column = self.ip_column + "_prefix"
        df[ip_prefix_column] = df[self.ip_column].apply(self._ip_to_prefix)

        df = df.drop(columns=self.ip_column)
        del self.search_keys[self.ip_column]
        del self.columns_renaming[self.ip_column]
        # self.search_keys[ipv6] = SearchKey.IPV6_ADDRESS
        self.search_keys[ip_binary] = SearchKey.IP_BINARY
        self.search_keys[ip_prefix_column] = SearchKey.IP_PREFIX
        # self.columns_renaming[ipv6] = original_ip
        self.columns_renaming[ip_binary] = original_ip
        self.columns_renaming[ip_prefix_column] = original_ip

        return df


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
