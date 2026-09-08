from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.auth import service as auth_service
from app.auth.api_keys import API_KEY_PREFIX, hash_api_key_secret
from app.auth.contracts import TrustedTenantContext
from app.auth.exceptions import (
    AuthBackendError,
    ExpiredApiCredentialError,
    InvalidApiCredentialError,
    RevokedApiCredentialError,
)

PEPPER = "backend-test-pepper"
KEY_ID = "K" * 22
SECRET = "s" * 43
RAW_KEY = f"{API_KEY_PREFIX}{KEY_ID}.{SECRET}"
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


@dataclass
class StubCredential:
    id: int = 101
    company_id: int = 202
    key_id: str = KEY_ID
    secret_hash: bytes = hash_api_key_secret(secret=SECRET, pepper=PEPPER)
    is_active: bool = True
    expires_at: datetime | None = None
    revoked_at: datetime | None = None


class RecordingSessionContext:
    def __init__(self):
        self.session = object()
        self.entered = False
        self.closed = False

    def __enter__(self):
        self.entered = True
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True
        return False


def _install_backend(monkeypatch, credential=None, error=None):
    contexts = []

    def session_factory():
        context = RecordingSessionContext()
        contexts.append(context)
        return context

    def find_credential(session, *, key_id):
        if error is not None:
            raise error
        assert key_id == KEY_ID
        return credential

    monkeypatch.setattr(auth_service, "SessionLocal", session_factory)
    monkeypatch.setattr(
        auth_service,
        "find_api_credential_by_key_id",
        find_credential,
    )
    monkeypatch.setattr(auth_service, "_utc_now", lambda: NOW)
    return contexts


def test_valid_credential_returns_frozen_context_from_database_identity(monkeypatch):
    credential = StubCredential(id=301, company_id=902)
    contexts = _install_backend(monkeypatch, credential)

    context = auth_service.authenticate_api_key(RAW_KEY, pepper=PEPPER)

    assert context == TrustedTenantContext(company_id=902, credential_id=301)
    assert context is not credential
    assert contexts[0].entered is True
    assert contexts[0].closed is True
    with pytest.raises(AttributeError):
        context.company_id = 1  # type: ignore[misc]


def test_verification_uses_compare_digest(monkeypatch):
    calls = []
    real_compare_digest = auth_service.hmac.compare_digest
    _install_backend(monkeypatch, StubCredential())

    def recording_compare_digest(left, right):
        calls.append((left, right))
        return real_compare_digest(left, right)

    monkeypatch.setattr(auth_service.hmac, "compare_digest", recording_compare_digest)

    auth_service.authenticate_api_key(RAW_KEY, pepper=PEPPER)

    assert len(calls) == 1
    assert all(isinstance(value, bytes) for value in calls[0])
    assert len(calls[0][0]) == len(calls[0][1]) == 32


def test_unknown_key_runs_dummy_constant_time_comparison(monkeypatch):
    calls = []
    real_compare_digest = auth_service.hmac.compare_digest
    contexts = _install_backend(monkeypatch, None)

    def recording_compare_digest(left, right):
        calls.append((left, right))
        return real_compare_digest(left, right)

    monkeypatch.setattr(auth_service.hmac, "compare_digest", recording_compare_digest)

    with pytest.raises(InvalidApiCredentialError, match="Invalid API credential"):
        auth_service.authenticate_api_key(RAW_KEY, pepper=PEPPER)

    assert len(calls) == 1
    assert calls[0][1] == bytes(32)
    assert contexts[0].closed is True


def test_wrong_secret_is_rejected_and_session_closes(monkeypatch):
    credential = StubCredential(
        secret_hash=hash_api_key_secret(secret="t" * 43, pepper=PEPPER)
    )
    contexts = _install_backend(monkeypatch, credential)

    with pytest.raises(InvalidApiCredentialError, match="Invalid API credential"):
        auth_service.authenticate_api_key(RAW_KEY, pepper=PEPPER)

    assert contexts[0].closed is True


@pytest.mark.parametrize(
    ("credential", "expected_error"),
    [
        (StubCredential(is_active=False), InvalidApiCredentialError),
        (
            StubCredential(is_active=False, revoked_at=NOW - timedelta(days=1)),
            RevokedApiCredentialError,
        ),
        (
            StubCredential(expires_at=NOW - timedelta(microseconds=1)),
            ExpiredApiCredentialError,
        ),
        (StubCredential(expires_at=NOW), ExpiredApiCredentialError),
    ],
)
def test_unavailable_credentials_are_rejected(monkeypatch, credential, expected_error):
    contexts = _install_backend(monkeypatch, credential)

    with pytest.raises(expected_error) as exc_info:
        auth_service.authenticate_api_key(RAW_KEY, pepper=PEPPER)

    assert str(exc_info.value) == "Invalid API credential"
    assert contexts[0].closed is True


def test_expiry_after_current_instant_is_valid(monkeypatch):
    credential = StubCredential(expires_at=NOW + timedelta(microseconds=1))
    _install_backend(monkeypatch, credential)

    assert auth_service.authenticate_api_key(RAW_KEY, pepper=PEPPER) == (
        TrustedTenantContext(company_id=202, credential_id=101)
    )


def test_malformed_key_is_rejected_before_opening_session(monkeypatch):
    contexts = _install_backend(monkeypatch, StubCredential())

    with pytest.raises(InvalidApiCredentialError, match="Invalid API credential"):
        auth_service.authenticate_api_key("malformed", pepper=PEPPER)

    assert contexts == []


def test_missing_pepper_is_safe_backend_failure_without_opening_session(monkeypatch):
    contexts = _install_backend(monkeypatch, StubCredential())

    with pytest.raises(AuthBackendError) as exc_info:
        auth_service.authenticate_api_key(RAW_KEY, pepper="")

    assert str(exc_info.value) == "Authentication backend unavailable"
    assert contexts == []


def test_database_failure_is_translated_and_session_closes(monkeypatch):
    contexts = _install_backend(
        monkeypatch,
        error=RuntimeError("postgresql://secret database internals"),
    )

    with pytest.raises(AuthBackendError) as exc_info:
        auth_service.authenticate_api_key(RAW_KEY, pepper=PEPPER)

    assert str(exc_info.value) == "Authentication backend unavailable"
    assert "postgresql" not in str(exc_info.value)
    assert contexts[0].closed is True


def test_session_factory_failure_is_translated(monkeypatch):
    def fail_session_factory():
        raise RuntimeError("database host details")

    monkeypatch.setattr(auth_service, "SessionLocal", fail_session_factory)

    with pytest.raises(AuthBackendError) as exc_info:
        auth_service.authenticate_api_key(RAW_KEY, pepper=PEPPER)

    assert str(exc_info.value) == "Authentication backend unavailable"


def test_sibling_and_cross_company_credentials_resolve_independently(monkeypatch):
    credentials = {}
    key_material = []
    for marker, credential_id, company_id in (
        ("A", 11, 100),
        ("B", 12, 100),
        ("C", 13, 200),
    ):
        key_id = marker * 22
        secret = marker.lower() * 43
        raw_key = f"{API_KEY_PREFIX}{key_id}.{secret}"
        credentials[key_id] = StubCredential(
            id=credential_id,
            company_id=company_id,
            key_id=key_id,
            secret_hash=hash_api_key_secret(secret=secret, pepper=PEPPER),
        )
        key_material.append((raw_key, credential_id, company_id))

    contexts = []

    def session_factory():
        context = RecordingSessionContext()
        contexts.append(context)
        return context

    monkeypatch.setattr(auth_service, "SessionLocal", session_factory)
    monkeypatch.setattr(
        auth_service,
        "find_api_credential_by_key_id",
        lambda session, *, key_id: credentials.get(key_id),
    )

    results = [
        auth_service.authenticate_api_key(raw_key, pepper=PEPPER)
        for raw_key, _, _ in key_material
    ]

    assert results == [
        TrustedTenantContext(company_id=company_id, credential_id=credential_id)
        for _, credential_id, company_id in key_material
    ]
    assert all(context.closed for context in contexts)


def test_revoking_one_sibling_does_not_invalidate_another(monkeypatch):
    active_key_id = "A" * 22
    revoked_key_id = "R" * 22
    active_secret = "a" * 43
    revoked_secret = "r" * 43
    credentials = {
        active_key_id: StubCredential(
            id=1,
            company_id=500,
            key_id=active_key_id,
            secret_hash=hash_api_key_secret(secret=active_secret, pepper=PEPPER),
        ),
        revoked_key_id: StubCredential(
            id=2,
            company_id=500,
            key_id=revoked_key_id,
            secret_hash=hash_api_key_secret(secret=revoked_secret, pepper=PEPPER),
            is_active=False,
            revoked_at=NOW,
        ),
    }
    monkeypatch.setattr(auth_service, "SessionLocal", RecordingSessionContext)
    monkeypatch.setattr(
        auth_service,
        "find_api_credential_by_key_id",
        lambda session, *, key_id: credentials.get(key_id),
    )

    with pytest.raises(RevokedApiCredentialError):
        auth_service.authenticate_api_key(
            f"{API_KEY_PREFIX}{revoked_key_id}.{revoked_secret}",
            pepper=PEPPER,
        )

    assert auth_service.authenticate_api_key(
        f"{API_KEY_PREFIX}{active_key_id}.{active_secret}",
        pepper=PEPPER,
    ) == TrustedTenantContext(company_id=500, credential_id=1)
