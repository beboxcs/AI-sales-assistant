"""Immutable data exchanged across the human-auth boundary."""

from dataclasses import dataclass
from datetime import datetime

from app.domain.enums import MembershipRole


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    user_id: int
    membership_id: int
    company_id: int
    role: MembershipRole


@dataclass(frozen=True)
class AuthenticatedUserContext:
    user_id: int
    membership_id: int
    company_id: int
    role: MembershipRole
    session_id: int


@dataclass(frozen=True)
class CreatedSession:
    token: str
    expires_at: datetime
    context: AuthenticatedUserContext
