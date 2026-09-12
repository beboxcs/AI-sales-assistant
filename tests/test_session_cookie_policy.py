from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie

import pytest
from fastapi import Response

from app.api import security as api_security


OPAQUE_SESSION_TOKEN = "opaque.session.token"


def test_session_cookie_policy_uses_frozen_transport_attributes(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    api_security.set_session_cookie(
        response,
        OPAQUE_SESSION_TOKEN,
        max_age=3600,
    )

    cookie = _session_cookie(response)
    assert _set_cookie_header(response).startswith("rea_session=")
    assert cookie.value == OPAQUE_SESSION_TOKEN
    assert cookie["httponly"] is True
    assert cookie["samesite"] == "Lax"
    assert cookie["path"] == "/"
    assert cookie["domain"] == ""
    assert cookie["max-age"] == "3600"


@pytest.mark.parametrize("app_env", ["local", "test"])
def test_session_cookie_is_not_secure_in_local_or_test(monkeypatch, app_env):
    monkeypatch.setattr(api_security.settings, "APP_ENV", app_env)
    response = Response()

    api_security.set_session_cookie(
        response,
        OPAQUE_SESSION_TOKEN,
        max_age=3600,
    )

    assert "Secure" not in _set_cookie_header(response)


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_session_cookie_is_secure_in_deployed_environments(monkeypatch, app_env):
    monkeypatch.setattr(api_security.settings, "APP_ENV", app_env)
    response = Response()

    api_security.set_session_cookie(
        response,
        OPAQUE_SESSION_TOKEN,
        max_age=3600,
    )

    assert _session_cookie(response)["secure"] is True


def test_clear_session_cookie_deletes_same_cookie_path_and_domain(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "production")
    response = Response()

    api_security.clear_session_cookie(response)

    cookie = _session_cookie(response)
    assert cookie.value == ""
    assert cookie["max-age"] == "0"
    assert cookie["path"] == "/"
    assert cookie["domain"] == ""
    assert cookie["samesite"] == "Lax"
    assert cookie["httponly"] is True
    assert cookie["secure"] is True


@pytest.mark.parametrize(
    "max_age",
    [
        api_security.SESSION_COOKIE_MAX_AGE_SECONDS,
        api_security.SESSION_COOKIE_MAX_AGE_SECONDS + 1,
        timedelta(hours=12),
    ],
)
def test_session_cookie_max_age_cannot_exceed_eight_hours(
    monkeypatch,
    max_age,
):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    api_security.set_session_cookie(
        response,
        OPAQUE_SESSION_TOKEN,
        max_age=max_age,
    )

    assert (
        _session_cookie(response)["max-age"]
        == str(api_security.SESSION_COOKIE_MAX_AGE_SECONDS)
    )


def test_session_cookie_expires_at_input_is_capped_to_eight_hours(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    api_security.set_session_cookie(
        response,
        OPAQUE_SESSION_TOKEN,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=12),
    )

    assert (
        _session_cookie(response)["max-age"]
        == str(api_security.SESSION_COOKIE_MAX_AGE_SECONDS)
    )


def test_session_cookie_expires_at_soon_controls_max_age(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    api_security.set_session_cookie(
        response,
        OPAQUE_SESSION_TOKEN,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    max_age = int(_session_cookie(response)["max-age"])
    assert 0 < max_age <= 3600


def test_session_cookie_expired_expires_at_uses_zero_max_age(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    api_security.set_session_cookie(
        response,
        OPAQUE_SESSION_TOKEN,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    assert _session_cookie(response)["max-age"] == "0"


def test_session_cookie_rejects_naive_expires_at(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    with pytest.raises(ValueError, match="expires_at must be timezone-aware"):
        api_security.set_session_cookie(
            response,
            OPAQUE_SESSION_TOKEN,
            expires_at=datetime.now(),
        )


def test_session_cookie_requires_expires_at_or_max_age(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    with pytest.raises(ValueError, match="Provide exactly one"):
        api_security.set_session_cookie(response, OPAQUE_SESSION_TOKEN)


def test_session_cookie_rejects_both_expires_at_and_max_age(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    with pytest.raises(ValueError, match="Provide exactly one"):
        api_security.set_session_cookie(
            response,
            OPAQUE_SESSION_TOKEN,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            max_age=60,
        )


def test_session_cookie_negative_max_age_uses_zero(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "local")
    response = Response()

    api_security.set_session_cookie(
        response,
        OPAQUE_SESSION_TOKEN,
        max_age=-1,
    )

    assert _session_cookie(response)["max-age"] == "0"


def test_session_cookie_value_remains_supplied_opaque_token(monkeypatch):
    monkeypatch.setattr(api_security.settings, "APP_ENV", "test")
    opaque_token = "tok_user_id-company_id-membership_id-role"
    response = Response()

    api_security.set_session_cookie(response, opaque_token, max_age=60)

    assert _session_cookie(response).value == opaque_token


def _set_cookie_header(response: Response) -> str:
    return response.headers["set-cookie"]


def _session_cookie(response: Response):
    parsed = SimpleCookie()
    parsed.load(_set_cookie_header(response))
    return parsed[api_security.SESSION_COOKIE_NAME]
