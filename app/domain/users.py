"""User identity domain helpers."""


def normalize_email(email: str) -> str:
    """Return the canonical email identity without provider-specific rewriting."""
    normalized_email = email.strip().casefold()
    if not normalized_email:
        raise ValueError("Email must not be blank")
    return normalized_email
