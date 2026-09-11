"""Opaque bearer-token primitives for server-side human sessions."""

import hashlib
import secrets

SESSION_TOKEN_ENTROPY_BYTES = 32


def generate_session_token() -> str:
    """Return a cryptographically secure opaque bearer credential."""
    return secrets.token_urlsafe(SESSION_TOKEN_ENTROPY_BYTES)


def digest_session_token(*, token: str) -> bytes:
    """Derive the non-reversible, deterministic database lookup digest."""
    return hashlib.sha256(token.encode("ascii")).digest()
