import json

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse

import logging

from upgini.http import __version__

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
            # if not ver.is_prerelease:  # TODO return after release
            version = max(version, ver)
    return version


def validate_version():
    try:
        current_version = parse(__version__)
        latest_version = get_version("upgini")
        if latest_version != current_version:
            msg = f"You use {current_version} version, but latest is {latest_version}"
            logging.warning(msg)
            print("WARNING: " + msg)
    except Exception as e:
        logging.exception(f"Failed to validate verion: {e}")
