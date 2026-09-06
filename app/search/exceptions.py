"""Exceptions exposed by backend search execution."""


class SearchExecutionError(Exception):
    """Base error for failures in the backend search execution boundary."""


class InvalidSearchScopeError(SearchExecutionError):
    """Raised when backend scope validation or resolution fails."""


class SearchScopeResolutionFailedError(SearchExecutionError):
    """Raised when scope resolution fails without exposing backend details."""


class SearchExecutionFailedError(SearchExecutionError):
    """Raised when search execution fails without exposing database errors."""
