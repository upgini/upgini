from typing import Any, Optional


class _ErrorMessages(object):
    not_none = '"{0}" should not be None.'
    not_none_or_empty = '"{0}" should not be None or empty.'


class HttpError(Exception):
    """Error from REST API."""

    def __init__(self, message, status_code):
        super(HttpError, self).__init__(message)
        self.status_code = status_code

    def __new__(cls, *args, **kwargs):
        if kwargs.get("status_code") == 401:
            cls = UnauthorizedError
        return Exception.__new__(cls, *args, **kwargs)


class UnauthorizedError(HttpError):
    """Unauthorized error from REST API."""

    def __init__(self, message, status_code):
        message = "Unauthorized, please check your authorization token ({})".format(message)
        super(UnauthorizedError, self).__init__(message, status_code)


def _not_none(param_name: str, param: Optional[Any]):
    if param is None:
        raise TypeError(_ErrorMessages.not_none.format(param_name))


def _not_none_or_empty(param_name: str, param: Optional[str]):
    if param is None:
        raise TypeError(_ErrorMessages.not_none_or_empty.format(param_name))
