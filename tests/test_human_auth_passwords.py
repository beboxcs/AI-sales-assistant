import pytest
from argon2 import PasswordHasher, Type
from argon2.exceptions import HashingError

from app.auth.human import AuthInfrastructureError
from app.auth.human import passwords as password_service
from app.auth.human.passwords import hash_password, verify_password


def test_hashing_produces_an_encoded_argon2id_hash_not_plaintext():
    password = "correct horse battery staple"

    password_hash = hash_password(password)

    assert password_hash.startswith("$argon2id$")
    assert password_hash != password
    assert password not in password_hash


def test_hashing_the_same_password_uses_independent_random_salts():
    first_hash = hash_password("same password")
    second_hash = hash_password("same password")

    assert first_hash != second_hash
    assert verify_password("same password", first_hash) is True
    assert verify_password("same password", second_hash) is True


def test_correct_password_verifies():
    password_hash = hash_password("valid password")

    assert verify_password("valid password", password_hash) is True


def test_incorrect_password_fails_verification():
    password_hash = hash_password("valid password")

    assert verify_password("wrong password", password_hash) is False


@pytest.mark.parametrize("password", ["", " ", "\t", "\n", " \t\n "])
def test_blank_password_cannot_be_hashed(password):
    with pytest.raises(ValueError, match="Password must not be blank"):
        hash_password(password)


@pytest.mark.parametrize(
    "password_hash",
    [
        "not-an-encoded-hash",
        "$2b$12$unsupportedbcryptvalue",
        "$argon2id$malformed",
        "",
        None,
    ],
)
def test_malformed_or_unsupported_hash_fails_closed(password_hash):
    assert verify_password("password", password_hash) is False


def test_malformed_hash_consumes_dummy_argon2_verification(monkeypatch):
    real_hasher = password_service._PASSWORD_HASHER
    verification_hashes = []

    class RecordingHasher:
        def verify(self, password_hash, password):
            verification_hashes.append(password_hash)
            return real_hasher.verify(password_hash, password)

    monkeypatch.setattr(password_service, "_PASSWORD_HASHER", RecordingHasher())

    assert verify_password("password", "$argon2id$malformed") is False
    assert verification_hashes == [
        "$argon2id$malformed",
        password_service._DUMMY_PASSWORD_HASH,
    ]


@pytest.mark.parametrize("hash_type", [Type.I, Type.D])
def test_non_argon2id_hashes_fail_closed(hash_type):
    password = "valid password"
    password_hash = PasswordHasher(type=hash_type).hash(password)

    assert verify_password(password, password_hash) is False


def test_unsupported_hash_consumes_only_dummy_argon2id_verification(monkeypatch):
    password = "valid password"
    unsupported_hash = PasswordHasher(type=Type.I).hash(password)
    real_hasher = password_service._PASSWORD_HASHER
    verification_hashes = []

    class RecordingHasher:
        def verify(self, password_hash, supplied_password):
            verification_hashes.append(password_hash)
            return real_hasher.verify(password_hash, supplied_password)

    monkeypatch.setattr(password_service, "_PASSWORD_HASHER", RecordingHasher())

    assert verify_password(password, unsupported_hash) is False
    assert verification_hashes == [password_service._DUMMY_PASSWORD_HASH]


def test_library_hashing_failure_is_translated_without_internal_details(
    monkeypatch,
):
    class FailingHasher:
        def hash(self, password):
            raise HashingError("library implementation details")

    monkeypatch.setattr(password_service, "_PASSWORD_HASHER", FailingHasher())

    with pytest.raises(AuthInfrastructureError) as exc_info:
        hash_password("valid password")

    assert str(exc_info.value) == "Password hashing unavailable"
    assert "implementation details" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
