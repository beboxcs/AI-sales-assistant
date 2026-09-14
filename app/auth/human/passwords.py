"""Backend-owned Argon2id password hashing primitives."""

from argon2 import PasswordHasher, Type
from argon2.exceptions import (
    HashingError,
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

from app.auth.human.exceptions import AuthInfrastructureError

_PASSWORD_HASHER = PasswordHasher(type=Type.ID)
_HASHING_ERROR_MESSAGE = "Password hashing unavailable"

# A non-secret Argon2id hash generated with the same PasswordHasher profile as
# real passwords. It keeps missing or unusable persisted hashes on an Argon2
# verification path without ever making those hashes acceptable.
_DUMMY_PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$YkMn+z/YWMb1U/L8S4AICw$"
    "KGGntZ7zvLmCzPdUMW2Sw1kXYM/sQfb7pN/ODvAYPNk"
)


def hash_password(password: str) -> str:
    """Hash a non-blank password with a library-generated random salt."""
    if not isinstance(password, str) or not password.strip():
        raise ValueError("Password must not be blank")
    try:
        return _PASSWORD_HASHER.hash(password)
    except HashingError:
        raise AuthInfrastructureError(_HASHING_ERROR_MESSAGE) from None


def verify_password(password: str, password_hash: str) -> bool:
    """Return whether a password matches an encoded Argon2id hash."""
    if not isinstance(password, str):
        return False
    if (
        not isinstance(password_hash, str)
        or not password_hash.startswith("$argon2id$")
    ):
        _perform_dummy_verification(password)
        return False
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except (InvalidHashError, VerificationError):
        _perform_dummy_verification(password)
        return False


def _perform_dummy_verification(password: str) -> None:
    try:
        _PASSWORD_HASHER.verify(_DUMMY_PASSWORD_HASH, password)
    except VerifyMismatchError:
        pass
