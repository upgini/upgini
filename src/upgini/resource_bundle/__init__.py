"""
Module containing the implementation of the ResourceBundle
"""
from __future__ import annotations

import codecs
import os
import re
from os import PathLike
from pathlib import Path
from typing import KeysView, Optional, Sequence

from .exceptions import MalformedResourceBundleError, NotInResourceBundleError

__version__ = "2.0.5"
__author__ = "Felix Zenk"
__email__ = "felix.zenk@web.de"


class _Parser:
    """
    A parser for the .properties file format.
    """

    @staticmethod
    def parse(file_path: Path) -> dict[str, str]:
        """
        Reads a ResourceBundle file and parses its contents

        :param Path file_path:
        :return: The contents of the file as a key-value dict
        """

        def is_comment(line: str) -> bool:
            return line.strip().startswith("#") or line.strip(" \n\r") == ""

        def decode(arg: str) -> str:
            # 'Unescape' chars
            pattern = re.compile(  # from https://stackoverflow.com/a/24519338/14563958
                r"""( \\U........  # 8-digit hex escapes
                | \\u....          # 4-digit hex escapes
                | \\x..            # 2-digit hex escapes
                | \\[0-7]{1,3}     # Octal escapes
                | \\N\{[^}]+\}     # Unicode characters by name
                | \\[\\'"abfnrtv]  # Single-character escapes
                )""",
                re.UNICODE | re.VERBOSE,
            )
            return re.sub(pattern, lambda match: codecs.decode(match.group(0), "unicode-escape"), arg)

        # I/O read
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # parse
        mapping = dict()
        enumerator = enumerate(lines, start=1)
        for line_no, line in enumerator:
            if is_comment(line):
                continue

            if line.strip().endswith("=\\"):  # single line continuation
                value = ""
                pair = line.split("=")
                key = pair[0].strip()
                new_value = "\\"  # init new_value
                while new_value.strip().endswith("\\"):  # more line continuations
                    new_line_no, new_value = next(enumerator)
                    value += new_value.strip().strip("\\")
            elif "=" in line:
                key, *value = line.split("=")
                key = key.strip()
                value = "=".join(value).strip()
            else:
                raise MalformedResourceBundleError(f"Malformed file: '{file_path}' (line {line_no})")

            if "\\" in value:  # may contain escaped chars
                value = decode(value)
            if key not in mapping.keys():
                mapping[key] = value
            else:
                raise MalformedResourceBundleError(f"Duplicate key '{key}' in file '{file_path}' on line {line_no}")
        return mapping


class ResourceBundle:
    """
    A ResourceBundle manages internationalization of string resources
    """

    __cached_bundles = dict()

    __slots__ = ["_name", "_locale", "_parent", "_mapping", "_path"]

    def __init__(self, bundle_name: str, bundle_locale: str | None, *, path: str | PathLike = None) -> None:
        """
        Initialize a ResourceBundle

        :param str bundle_name:
            The base name of the ResourceBundle
        :param str | None bundle_locale:
            The specific locale consisting of a ISO 639-1 language code,
            a script code, an ISO 3166-1 alpha-2 country code and a variant code
            separated by underscores.
            The format for a locale therefore is 'language_script_country_variant'
            like 'en_Latn_US_WINDOWS' or parts of it like 'en_US'
        :param str | PathLike | None path:
            The path to the ResourceBundle .properties files
            (default: current working directory)
        """
        self._name: str = bundle_name
        self._locale: str | None = bundle_locale
        self._path: Path = self._ensure_path(path)
        self._parent: ResourceBundle | None = self._get_parent_bundle()
        self._mapping: dict[str, str] = self._map()
        # Save self in cache
        self.__cached_bundles[self.name] = self

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        """
        Get the full name of the ResourceBundle
        """
        return f"{self._name}" if self._locale is None else f"{self._name}_{self._locale}"

    @property
    def parent(self) -> ResourceBundle:
        """
        Get the ResourceBundles parent ResourceBundle
        """
        return self._parent

    def _get_parent_bundle(self) -> ResourceBundle | None:
        # This is the root bundle
        if self._locale is None:
            return None

        # Cut one part off of the locale
        *parts, _ = self._locale.split("_")
        # If nothing is left set the locale to None else to the shortened locale
        parent_locale = "_".join(parts) if len(parts) > 0 else None
        # If cached
        if self.__cached_bundles.get(parent_locale) is not None:
            return self.__cached_bundles.get(parent_locale)
        # Not cached, start building chain
        return ResourceBundle(self._name, parent_locale, path=self._path)

    def _map(self) -> dict[str, str]:
        try:
            return _Parser.parse(self._path / f"{self.name}.properties")
        except FileNotFoundError:
            return dict()

    @staticmethod
    def _ensure_path(path: str | PathLike | None) -> Path:
        if isinstance(path, PathLike):
            return Path(path.__fspath__())
        if isinstance(path, str):
            return Path(path)
        if path is None:
            return Path()
        raise TypeError(f"Path must be of type str or PathLike, not {type(path)}")

    def __getitem__(self, item) -> str:
        return self.get(item)

    def keys(self) -> KeysView[str, str]:
        """
        Return all keys present in this specific ResourceBundle
        """
        return self._mapping.keys()

    def get(self, item: str, __default: str = None) -> str:
        """
        Get the value of ``item`` from this or a parent ResourceBundle

        :param str item: The key to look up
        :param str __default: A default value that is returned if the key can't be found anywhere
        :return: The value
        """
        # found in mapping
        if self._mapping.get(item) is not None:
            return self._mapping.get(item)

        # ask parent
        if self._parent is not None:
            return self._parent.get(item)

        # is root bundle (has no parent)
        if __default is not None:
            return __default

        raise NotInResourceBundleError(self.name, item)


def get_bundle(bundle_name: str, locale: str | Sequence[str] = None, path: Path | str = None) -> ResourceBundle:
    """
    Return a new :class:`ResourceBundle` after parsing the locale

    :param str bundle_name: The base name of the ResourceBundle
    :param str | Sequence[str | str] locale: The locale as a string or from the locale module
    :param Path | str | None path: The path to the ResourceBundle .properties files (default: current working directory)
    """
    # locale was not supplied
    if locale is None:
        return ResourceBundle(bundle_name=bundle_name, bundle_locale=None, path=path)

    # simple string
    if isinstance(locale, str):
        return ResourceBundle(bundle_name=bundle_name, bundle_locale=locale, path=path)

    # locale from the locale module
    extracted_locale, _ = locale
    return ResourceBundle(bundle_name=bundle_name, bundle_locale=extracted_locale, path=path)


bundle = ResourceBundle("strings", None, path=os.path.dirname(os.path.realpath(__file__)))
custom_bundles = dict()


def get_custom_bundle(custom_cfg: Optional[str] = None) -> ResourceBundle:
    global custom_bundles
    if custom_cfg is not None:
        custom_bundle = custom_bundles.get(custom_cfg)
        if custom_bundle is None:
            custom_bundle = ResourceBundle("strings", custom_cfg, path=os.path.dirname(os.path.realpath(__file__)))
            custom_bundles[custom_cfg] = custom_bundle
        return custom_bundle
    else:
        return bundle
