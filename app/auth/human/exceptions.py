"""Public exceptions raised by the human-auth boundary."""


class HumanAuthError(Exception):
    """Base error for human authentication operations."""


class InvalidCredentialsError(HumanAuthError):
    """Raised when supplied human credentials are invalid."""


class InvalidCsrfTokenError(HumanAuthError):
    """Raised when a CSRF token cannot be validated."""


class LoginRateLimitExceededError(HumanAuthError):
    """Raised when a login attempt exceeds a rate limit."""

    retry_after_seconds: int

    def __init__(self, retry_after_seconds: int):
        super().__init__("Login rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


class LoginRateLimitInfrastructureError(HumanAuthError):
    """Raised when login rate limiting cannot safely complete."""


class InvalidSessionError(HumanAuthError):
    """Raised when a session cannot establish an authenticated context."""


class SessionExpiredError(InvalidSessionError):
    """Raised when a session has expired."""


class SessionRevokedError(InvalidSessionError):
    """Raised when a session has been revoked."""


class InactiveUserError(InvalidSessionError):
    """Raised when the user associated with a session is inactive."""


class InactiveMembershipError(InvalidSessionError):
    """Raised when the membership associated with a session is inactive."""


class AuthInfrastructureError(HumanAuthError):
    """Raised when human-auth infrastructure cannot complete an operation."""
