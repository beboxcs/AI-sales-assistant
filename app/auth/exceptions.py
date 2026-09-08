class AuthenticationError(Exception):
    """Base error for the backend authentication boundary."""


class InvalidApiCredentialError(AuthenticationError):
    """Raised when API-key authentication cannot establish trusted identity."""


class RevokedApiCredentialError(InvalidApiCredentialError):
    """Raised when an otherwise valid credential has been revoked."""


class ExpiredApiCredentialError(InvalidApiCredentialError):
    """Raised when an otherwise valid credential has expired."""


class AuthBackendError(AuthenticationError):
    """Raised when authentication infrastructure cannot complete the request."""
