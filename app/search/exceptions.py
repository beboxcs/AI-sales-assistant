"""Exceptions exposed by backend search execution."""


class SearchExecutionError(Exception):
    """Base error for failures in the backend search execution boundary."""


class InvalidSearchScopeError(SearchExecutionError):
    """Raised when backend scope validation or resolution fails."""


class SearchExecutionFailedError(SearchExecutionError):
    """Raised when search execution fails without exposing database errors."""
