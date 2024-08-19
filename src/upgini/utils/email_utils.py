import logging
import re
from hashlib import sha256
from typing import Dict, List, Optional

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from upgini.metadata import SearchKey
from upgini.resource_bundle import ResourceBundle, get_custom_bundle
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


class EmailDomainGenerator:
    DOMAIN_SUFFIX = "_domain"

    def __init__(self, email_columns: List[str]):
        self.email_columns = email_columns
        self.generated_features = []

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        for email_col in self.email_columns:
            domain_feature = email_col + self.DOMAIN_SUFFIX
            df[domain_feature] = df[email_col].apply(self._email_to_domain)
            self.generated_features.append(domain_feature)
        return df

    @staticmethod
    def _email_to_domain(email: str) -> Optional[str]:
        if email is not None and isinstance(email, str) and "@" in email:
            name_and_domain = email.split("@")
            if len(name_and_domain) == 2 and len(name_and_domain[1]) > 0:
                return name_and_domain[1]


class EmailSearchKeyConverter:
    HEM_SUFFIX = "_hem"
    ONE_DOMAIN_SUFFIX = "_one_domain"

    def __init__(
        self,
        email_column: str,
        hem_column: Optional[str],
        search_keys: Dict[str, SearchKey],
        columns_renaming: Dict[str, str],
        unnest_search_keys: Optional[List[str]] = None,
        bundle: Optional[ResourceBundle] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.email_column = email_column
        self.hem_column = hem_column
        self.search_keys = search_keys
        self.columns_renaming = columns_renaming
        self.unnest_search_keys = unnest_search_keys
        self.bundle = bundle or get_custom_bundle()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")
        self.email_converted_to_hem = False

    @staticmethod
    def _email_to_hem(email: str) -> Optional[str]:
        if email is None or not isinstance(email, str) or email == "":
            return None

        if not EMAIL_REGEX.fullmatch(email):
            return None

        return sha256(email.lower().encode("utf-8")).hexdigest().lower()

    @staticmethod
    def _email_to_one_domain(email: str) -> Optional[str]:
        if email is not None and isinstance(email, str) and "@" in email:
            name_and_domain = email.split("@")
            if len(name_and_domain) == 2 and len(name_and_domain[0]) > 0 and len(name_and_domain[1]) > 0:
                return name_and_domain[0][0] + name_and_domain[1]

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        original_email_column = self.columns_renaming[self.email_column]
        if self.hem_column is None:
            hem_name = self.email_column + self.HEM_SUFFIX
            df[hem_name] = df[self.email_column].apply(self._email_to_hem)
            if df[hem_name].isna().all():
                msg = self.bundle.get("all_emails_invalid").format(self.email_column)
                print(msg)
                self.logger.warning(msg)
                df = df.drop(columns=hem_name)
                del self.search_keys[self.email_column]
                return df
            self.search_keys[hem_name] = SearchKey.HEM
            if self.email_column in self.unnest_search_keys:
                self.unnest_search_keys.append(hem_name)
            self.columns_renaming[hem_name] = original_email_column  # it could be upgini_email_unnest...
            self.email_converted_to_hem = True
        else:
            df[self.hem_column] = df[self.hem_column].astype("string").str.lower()

        del self.search_keys[self.email_column]
        if self.email_column in self.unnest_search_keys:
            self.unnest_search_keys.remove(self.email_column)

        one_domain_name = self.email_column + self.ONE_DOMAIN_SUFFIX
        df[one_domain_name] = df[self.email_column].apply(self._email_to_one_domain)
        self.columns_renaming[one_domain_name] = original_email_column
        self.search_keys[one_domain_name] = SearchKey.EMAIL_ONE_DOMAIN

        if self.email_converted_to_hem:
            df = df.drop(columns=self.email_column)
            del self.columns_renaming[self.email_column]

        return df
