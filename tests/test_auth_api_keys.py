import inspect

import pytest

from app.auth.api_keys import (
    API_KEY_PREFIX,
    KEY_ID_ENTROPY_BYTES,
    MAX_API_KEY_LENGTH,
    MAX_KEY_ID_LENGTH,
    MAX_SECRET_LENGTH,
    MIN_KEY_ID_LENGTH,
    MIN_SECRET_LENGTH,
    SECRET_ENTROPY_BYTES,
    generate_api_key,
    hash_api_key_secret,
    parse_api_key,
)
from app.auth.exceptions import InvalidApiCredentialError

VALID_KEY_ID = "K" * 22
VALID_SECRET = "s" * 43
VALID_RAW_KEY = f"{API_KEY_PREFIX}{VALID_KEY_ID}.{VALID_SECRET}"


def test_parse_api_key_returns_only_public_key_id_and_secret():
    assert parse_api_key(VALID_RAW_KEY) == (VALID_KEY_ID, VALID_SECRET)


@pytest.mark.parametrize(
    "raw_key",
    [
        "",
        f"rea_v2_{VALID_KEY_ID}.{VALID_SECRET}",
        f"rea_{VALID_KEY_ID}.{VALID_SECRET}",
        f"{API_KEY_PREFIX}.{VALID_SECRET}",
        f"{API_KEY_PREFIX}{VALID_KEY_ID}.",
        f"{API_KEY_PREFIX}{VALID_KEY_ID}.{VALID_SECRET}.extra",
        f"{API_KEY_PREFIX}{'K' * (MIN_KEY_ID_LENGTH - 1)}.{VALID_SECRET}",
        f"{API_KEY_PREFIX}{'K' * (MAX_KEY_ID_LENGTH + 1)}.{VALID_SECRET}",
        f"{API_KEY_PREFIX}{VALID_KEY_ID}.{'s' * (MIN_SECRET_LENGTH - 1)}",
        f"{API_KEY_PREFIX}{VALID_KEY_ID}.{'s' * (MAX_SECRET_LENGTH + 1)}",
        f"{API_KEY_PREFIX}unsafe$keyid.{VALID_SECRET}",
        f"{API_KEY_PREFIX}{VALID_KEY_ID}.unsafe/secret",
        "x" * (MAX_API_KEY_LENGTH + 1),
    ],
)
def test_parse_api_key_rejects_malformed_or_unbounded_input(raw_key):
    with pytest.raises(InvalidApiCredentialError) as exc_info:
        parse_api_key(raw_key)

    assert str(exc_info.value) == "Invalid API credential"
    if raw_key:
        assert raw_key not in str(exc_info.value)


def test_parse_api_key_rejects_non_string_input():
    with pytest.raises(InvalidApiCredentialError, match="Invalid API credential"):
        parse_api_key(None)  # type: ignore[arg-type]


def test_generate_api_key_uses_opaque_random_url_safe_components():
    generated_keys = {generate_api_key() for _ in range(50)}

    assert len(generated_keys) == 50
    assert inspect.signature(generate_api_key).parameters == {}
    for raw_key in generated_keys:
        key_id, secret = parse_api_key(raw_key)
        assert raw_key.startswith(API_KEY_PREFIX)
        assert MIN_KEY_ID_LENGTH <= len(key_id) <= MAX_KEY_ID_LENGTH
        assert MIN_SECRET_LENGTH <= len(secret) <= MAX_SECRET_LENGTH
        assert key_id.replace("-", "").replace("_", "").isalnum()
        assert secret.replace("-", "").replace("_", "").isalnum()


def test_generation_entropy_configuration_exceeds_frozen_minimums():
    assert KEY_ID_ENTROPY_BYTES * 8 >= 72
    assert SECRET_ENTROPY_BYTES * 8 >= 256


def test_hmac_digest_is_deterministic_and_sensitive_to_secret_and_pepper():
    digest = hash_api_key_secret(secret=VALID_SECRET, pepper="pepper-one")

    assert len(digest) == 32
    assert digest == hash_api_key_secret(
        secret=VALID_SECRET,
        pepper="pepper-one",
    )
    assert digest != hash_api_key_secret(
        secret="t" * 43,
        pepper="pepper-one",
    )
    assert digest != hash_api_key_secret(
        secret=VALID_SECRET,
        pepper="pepper-two",
    )
    assert digest != VALID_SECRET.encode("ascii")
