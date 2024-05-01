import logging
import re
from hashlib import sha256
from typing import Dict, List, Optional

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from upgini.metadata import SearchKey
from upgini.resource_bundle import bundle
from upgini.utils.base_search_key_detector import BaseSearchKeyDetector

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9.!#$%&’*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$")


class EmailSearchKeyDetector(BaseSearchKeyDetector):
    def _is_search_key_by_name(self, column_name: str) -> bool:
        return str(column_name).lower() in ["email", "e_mail", "e-mail"]

    def _is_search_key_by_values(self, column: pd.Series) -> bool:
        if not is_string_dtype(column) and not is_object_dtype:
            return False
        if not column.astype("string").str.contains("@").any():
            return False

        all_count = len(column)
        is_email_count = len(column.loc[column.astype("string").str.fullmatch(EMAIL_REGEX)])
        return is_email_count / all_count > 0.1


class EmailSearchKeyConverter:
    HEM_COLUMN_NAME = "hashed_email"
    DOMAIN_COLUMN_NAME = "email_domain"
    EMAIL_ONE_DOMAIN_COLUMN_NAME = "email_one_domain"

    def __init__(
        self,
        email_column: str,
        hem_column: Optional[str],
        search_keys: Dict[str, SearchKey],
        unnest_search_keys: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.email_column = email_column
        self.hem_column = hem_column
        self.search_keys = search_keys
        self.unnest_search_keys = unnest_search_keys
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")
        self.generated_features: List[str] = []
        self.email_converted_to_hem = False

    @staticmethod
    def _email_to_hem(email: str) -> Optional[str]:
        if email is None or not isinstance(email, str) or email == "":
            return None

        if not EMAIL_REGEX.fullmatch(email):
            return None

        return sha256(email.lower().encode("utf-8")).hexdigest()

    @staticmethod
    def _email_to_one_domain(email: str) -> Optional[str]:
        if email is not None and isinstance(email, str) and "@" in email:
            name_and_domain = email.split("@")
            if len(name_and_domain) == 2 and len(name_and_domain[0]) > 0 and len(name_and_domain[1]) > 0:
                return name_and_domain[0][0] + name_and_domain[1]

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.hem_column is None:
            df[self.HEM_COLUMN_NAME] = df[self.email_column].apply(self._email_to_hem)
            if df[self.HEM_COLUMN_NAME].isna().all():
                msg = bundle.get("all_emails_invalid").format(self.email_column)
                print(msg)
                self.logger.warning(msg)
                df = df.drop(columns=self.HEM_COLUMN_NAME)
                del self.search_keys[self.email_column]
                return df
            self.search_keys[self.HEM_COLUMN_NAME] = SearchKey.HEM
            self.unnest_search_keys.append(self.HEM_COLUMN_NAME)
            self.email_converted_to_hem = True

        del self.search_keys[self.email_column]
        if self.email_column in self.unnest_search_keys:
            self.unnest_search_keys.remove(self.email_column)

        df[self.EMAIL_ONE_DOMAIN_COLUMN_NAME] = df[self.email_column].apply(self._email_to_one_domain)

        self.search_keys[self.EMAIL_ONE_DOMAIN_COLUMN_NAME] = SearchKey.EMAIL_ONE_DOMAIN

        df[self.DOMAIN_COLUMN_NAME] = df[self.EMAIL_ONE_DOMAIN_COLUMN_NAME].str[1:]
        self.generated_features.append(self.DOMAIN_COLUMN_NAME)

        return df
