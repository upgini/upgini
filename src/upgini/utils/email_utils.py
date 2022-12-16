import logging
import re
from hashlib import sha256
from typing import Dict, List, Optional

import pandas as pd
from pandas.api.types import is_string_dtype

from upgini.metadata import SearchKey
from upgini.utils.base_search_key_detector import BaseSearchKeyDetector


EMAIL_REGEX = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')


class EmailSearchKeyDetector(BaseSearchKeyDetector):
    def _is_search_key_by_name(self, column_name: str) -> bool:
        return str(column_name).lower() in ["email", "e_mail", "e-mail"]

    def _is_search_key_by_values(self, column: pd.Series) -> bool:
        if not is_string_dtype(column):
            return False

        all_count = len(column)
        is_email_count = len(
            column.loc[
                column.astype("string").str.fullmatch(EMAIL_REGEX)
            ]
        )
        return is_email_count / all_count > 0.1


class EmailSearchKeyConverter:

    HEM_COLUMN_NAME = "hashed_email"
    DOMAIN_COLUMN_NAME = "email_domain"

    def __init__(
        self,
        email_column: str,
        hem_column: Optional[str],
        search_keys: Dict[str, SearchKey],
        logger: Optional[logging.Logger] = None,
    ):
        self.email_column = email_column
        self.hem_column = hem_column
        self.search_keys = search_keys
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")
        self.generated_features: List[str] = []

    @staticmethod
    def _email_to_hem(email: str) -> Optional[str]:
        if email is None or not isinstance(email, str) or email == "":
            return None

        if not EMAIL_REGEX.fullmatch(email):
            return None

        return sha256(email.lower().encode("utf-8")).hexdigest()

    @staticmethod
    def _email_to_domain(email: str) -> Optional[str]:
        if email is not None and type(email) == str and "@" in email:
            domain_candidate = email.split("@")[1]
            if len(domain_candidate) > 0:
                return domain_candidate

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.hem_column is None:
            df[self.HEM_COLUMN_NAME] = df[self.email_column].apply(self._email_to_hem)
            self.search_keys[self.HEM_COLUMN_NAME] = SearchKey.HEM

        del self.search_keys[self.email_column]

        df[self.DOMAIN_COLUMN_NAME] = df[self.email_column].apply(self._email_to_domain)
        self.generated_features.append(self.DOMAIN_COLUMN_NAME)
        df.drop(columns=self.email_column, inplace=True)

        return df
