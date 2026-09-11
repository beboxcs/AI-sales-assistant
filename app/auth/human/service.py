"""Human-auth service boundary.

Authentication establishes trusted identity. Authorization remains a separate
layer and must decide what an authenticated principal may do. Human identity
is derived only from persistence and remains separate from B1 API-key tenant
identity.
"""

from app.auth.human.exceptions import (
    AuthInfrastructureError,
    HumanAuthError,
    InvalidCredentialsError,
)
from app.auth.human.passwords import verify_password
from app.auth.human.repository import (
    find_active_memberships_for_user,
    find_user_by_normalized_email,
)
from app.auth.human.types import (
    AuthenticatedPrincipal,
    AuthenticatedUserContext,
    CreatedSession,
)
from app.database import SessionLocal
from app.domain.enums import MembershipRole
from app.domain.users import normalize_email

_INVALID_CREDENTIALS_MESSAGE = "Invalid credentials"
_INFRASTRUCTURE_ERROR_MESSAGE = "Human authentication unavailable"

# A non-secret Argon2id hash generated with the same PasswordHasher profile as
# real passwords. Unknown users are verified against it so email existence does
# not skip the expensive password-verification path.
_DUMMY_PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$YkMn+z/YWMb1U/L8S4AICw$"
    "KGGntZ7zvLmCzPdUMW2Sw1kXYM/sQfb7pN/ODvAYPNk"
)


def authenticate_credentials(
    email: str,
    password: str,
) -> AuthenticatedPrincipal:
    try:
        try:
            normalized_email = normalize_email(email)
        except (AttributeError, ValueError):
            verify_password(password, _DUMMY_PASSWORD_HASH)
            raise InvalidCredentialsError(
                _INVALID_CREDENTIALS_MESSAGE
            ) from None

        with SessionLocal() as session:
            user = find_user_by_normalized_email(
                session,
                normalized_email=normalized_email,
            )
            password_hash = (
                user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
            )
            password_matches = verify_password(password, password_hash)

            if user is None or user.is_active is not True or not password_matches:
                raise InvalidCredentialsError(_INVALID_CREDENTIALS_MESSAGE)

            active_memberships = [
                membership
                for membership in find_active_memberships_for_user(
                    session,
                    user_id=user.id,
                )
                if membership.is_active is True
            ]
            if len(active_memberships) != 1:
                raise InvalidCredentialsError(_INVALID_CREDENTIALS_MESSAGE)

            membership = active_memberships[0]
            try:
                role = MembershipRole(membership.role)
            except (TypeError, ValueError) as exc:
                raise AuthInfrastructureError(
                    _INFRASTRUCTURE_ERROR_MESSAGE
                ) from exc

            return AuthenticatedPrincipal(
                user_id=user.id,
                membership_id=membership.id,
                company_id=membership.company_id,
                role=role,
            )
    except HumanAuthError:
        raise
    except Exception as exc:
        raise AuthInfrastructureError(_INFRASTRUCTURE_ERROR_MESSAGE) from exc


def create_session(
    *,
    principal: AuthenticatedPrincipal,
) -> CreatedSession:
    raise NotImplementedError


def authenticate_session(token: str) -> AuthenticatedUserContext:
    raise NotImplementedError


def revoke_session(session_id: int) -> None:
    raise NotImplementedError
