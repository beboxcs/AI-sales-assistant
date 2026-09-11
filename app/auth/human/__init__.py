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
    InvalidSessionError,
    SessionExpiredError,
    SessionRevokedError,
)
from app.auth.human.service import (
    authenticate_credentials,
    authenticate_session,
    create_session,
    revoke_session,
)
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)

__all__ = [
    "AuthenticatedPrincipal",
    "AuthenticatedUserContext",
    "CreatedSession",
    "HumanAuthError",
    "InvalidCredentialsError",
    "InvalidSessionError",
    "SessionExpiredError",
    "SessionRevokedError",
    "InactiveUserError",
    "InactiveMembershipError",
    "AuthInfrastructureError",
    "authenticate_credentials",
    "create_session",
    "authenticate_session",
    "revoke_session",
]
