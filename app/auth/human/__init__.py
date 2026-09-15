"""Human-auth public contract.

Authentication establishes trusted identity. Authorization is a separate
layer that consumes this identity and determines permitted actions.
"""

from app.auth.human.exceptions import (
    AuthInfrastructureError,
    HumanAuthError,
    InactiveMembershipError,
    InactiveUserError,
    InvalidCredentialsError,
    InvalidCsrfTokenError,
    InvalidSessionError,
    LoginRateLimitExceededError,
    LoginRateLimitInfrastructureError,
    SessionExpiredError,
    SessionRevokedError,
)
from app.auth.human.service import (
    authenticate_credentials,
    create_session,
    issue_csrf_token,
    resolve_session,
    revoke_session,
    verify_csrf_token,
)
from app.auth.human.login_rate_limit import (
    check_login_rate_limit,
    record_login_aborted,
    record_login_failure,
    record_login_success,
)
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
    LoginRateLimitPermit,
)

__all__ = [
    "AuthenticatedPrincipal",
    "AuthenticatedUserContext",
    "CreatedSession",
    "LoginRateLimitPermit",
    "HumanAuthError",
    "InvalidCredentialsError",
    "InvalidCsrfTokenError",
    "InvalidSessionError",
    "LoginRateLimitExceededError",
    "LoginRateLimitInfrastructureError",
    "SessionExpiredError",
    "SessionRevokedError",
    "InactiveUserError",
    "InactiveMembershipError",
    "AuthInfrastructureError",
    "authenticate_credentials",
    "create_session",
    "issue_csrf_token",
    "resolve_session",
    "revoke_session",
    "verify_csrf_token",
    "check_login_rate_limit",
    "record_login_aborted",
    "record_login_failure",
    "record_login_success",
]
