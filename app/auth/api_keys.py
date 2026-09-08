import hashlib
import hmac
import re
import secrets

from app.auth.exceptions import InvalidApiCredentialError

API_KEY_PREFIX = "rea_v1_"
KEY_ID_ENTROPY_BYTES = 16
SECRET_ENTROPY_BYTES = 32
MIN_KEY_ID_LENGTH = 20
MAX_KEY_ID_LENGTH = 64
MIN_SECRET_LENGTH = 43
MAX_SECRET_LENGTH = 128
MAX_API_KEY_LENGTH = len(API_KEY_PREFIX) + MAX_KEY_ID_LENGTH + 1 + MAX_SECRET_LENGTH

_URL_SAFE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_INVALID_CREDENTIAL_MESSAGE = "Invalid API credential"


def generate_api_key() -> str:
    key_id = secrets.token_urlsafe(KEY_ID_ENTROPY_BYTES)
    secret = secrets.token_urlsafe(SECRET_ENTROPY_BYTES)
    return f"{API_KEY_PREFIX}{key_id}.{secret}"


def parse_api_key(raw_key: str) -> tuple[str, str]:
    if not isinstance(raw_key, str) or len(raw_key) > MAX_API_KEY_LENGTH:
        raise InvalidApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)
    if not raw_key.startswith(API_KEY_PREFIX) or raw_key.count(".") != 1:
        raise InvalidApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)

    public_part, secret = raw_key.split(".", maxsplit=1)
    key_id = public_part[len(API_KEY_PREFIX):]
    if not MIN_KEY_ID_LENGTH <= len(key_id) <= MAX_KEY_ID_LENGTH:
        raise InvalidApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)
    if not MIN_SECRET_LENGTH <= len(secret) <= MAX_SECRET_LENGTH:
        raise InvalidApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)
    if _URL_SAFE_TOKEN_PATTERN.fullmatch(key_id) is None:
        raise InvalidApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)
    if _URL_SAFE_TOKEN_PATTERN.fullmatch(secret) is None:
        raise InvalidApiCredentialError(_INVALID_CREDENTIAL_MESSAGE)
    return key_id, secret


def hash_api_key_secret(*, secret: str, pepper: str) -> bytes:
    return hmac.new(
        pepper.encode("utf-8"),
        secret.encode("ascii"),
        hashlib.sha256,
    ).digest()
