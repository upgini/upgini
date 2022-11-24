class ResourceBundleError(LookupError):
    pass


class MalformedResourceBundleError(ResourceBundleError):
    """
    Error that indicates that a ResourceBundle is malformed.
    """


class NotInResourceBundleError(ResourceBundleError):

    def __init__(self, bundle_name: str, key: str):
        """
        Error that is raised when a key could not be found in a ResourceBundle.

        :param str bundle_name: The name of the ResourceBundle
        :param str key: The key that could not be found
        """
        super(NotInResourceBundleError, self).__init__(f"Can't find key {key} in bundle {bundle_name}!")
