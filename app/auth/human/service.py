"""Frozen human-auth service signatures.

Authentication establishes trusted identity. Authorization remains a separate
layer and must decide what an authenticated principal may do.
"""

from datetime import datetime

from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)


def authenticate_credentials(
    email: str,
    password: str,
) -> AuthenticatedPrincipal:
    raise NotImplementedError


def create_session(
    principal: AuthenticatedPrincipal,
    expires_at: datetime,
) -> CreatedSession:
    raise NotImplementedError


def authenticate_session(token: str) -> AuthenticatedUserContext:
    raise NotImplementedError


def revoke_session(session_id: int) -> None:
    raise NotImplementedError
