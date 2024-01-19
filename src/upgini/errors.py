class HttpError(Exception):
    """Error from REST API."""

    def __init__(self, message: str, status_code):
        super(HttpError, self).__init__(message)
        self.status_code = status_code
        self.message = message

    def __new__(cls, *args, **kwargs):
        if kwargs.get("status_code") == 401:
            cls = UnauthorizedError
        return Exception.__new__(cls, *args, **kwargs)


class UnauthorizedError(HttpError):
    """Unauthorized error from REST API."""

    def __init__(self, message, status_code):
        message = "Unauthorized, please check your authorization token ({})".format(message)
        super(UnauthorizedError, self).__init__(message, status_code)


class UpginiConnectionError(Exception):
    def __init__(self, message):
        super(UpginiConnectionError, self).__init__(message)


class ValidationError(Exception):
    def __init__(self, message):
        super(ValidationError, self).__init__(message)
