"""Frozen human-auth service signatures.

Authentication establishes trusted identity. Authorization remains a separate
layer and must decide what an authenticated principal may do.
"""

from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)


def authenticate_credentials(
    *,
    email: str,
    password: str,
) -> AuthenticatedPrincipal:
    raise NotImplementedError


def create_session(
    *,
    principal: AuthenticatedPrincipal,
) -> CreatedSession:
    raise NotImplementedError


def resolve_session(
    *,
    token: str,
) -> AuthenticatedUserContext:
    raise NotImplementedError


def revoke_session(
    *,
    token: str,
) -> None:
    raise NotImplementedError
