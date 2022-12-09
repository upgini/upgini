import json

import requests
import threading

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse

import logging

from upgini.http import __version__
from upgini.resource_bundle import bundle

URL_PATTERN = "https://pypi.python.org/pypi/{package}/json"


def get_version(package, url_pattern=URL_PATTERN):
    """Return version of package on pypi.python.org using json."""
    req = requests.get(url_pattern.format(package=package))
    version = parse("0")
    if req.status_code == requests.codes.ok:
        j = json.loads(req.text.encode(req.encoding or "utf8"))
        releases = j.get("releases", [])
        for release in releases:
            ver = parse(release)
            if not ver.is_prerelease:
                version = max(version, ver)
    return version


def validate_version(logger: logging.Logger):
    def task():
        try:
            current_version = parse(__version__)
            latest_version = get_version("upgini")
            if current_version < latest_version:  # type: ignore
                msg = bundle.get("version_warning").format(current_version, latest_version)
                logger.warning(msg)
                print(msg)
        except Exception:
            logger.warning("Failed to validate version", exc_info=True)

    thread = threading.Thread(target=task)
    thread.start()
