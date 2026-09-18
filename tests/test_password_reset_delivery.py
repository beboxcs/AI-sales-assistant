from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.api import password_reset_delivery
from app.api.password_reset_delivery import (
    DisabledPasswordResetDeliveryService,
    PasswordResetDeliveryError,
    SmtpPasswordResetDeliveryService,
)
from app.config import Settings


EXPIRES_AT = datetime(2026, 9, 18, 12, 30, tzinfo=timezone.utc)


def test_smtp_password_reset_delivery_builds_safe_encoded_url(monkeypatch):
    sent_messages = []

    class FakeSmtp:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def send_message(self, message):
            sent_messages.append((self.host, self.port, self.timeout, message))

    monkeypatch.setattr(password_reset_delivery.smtplib, "SMTP", FakeSmtp)
    service = SmtpPasswordResetDeliveryService(
        host="mailpit",
        port=1025,
        from_email="security@example.test",
        reset_url="http://localhost:3000/reset-password?source=email",
    )

    service.deliver_password_reset(
        recipient_email="reset@example.com",
        token="token with spaces&symbols",
        expires_at=EXPIRES_AT,
    )

    assert len(sent_messages) == 1
    host, port, timeout, message = sent_messages[0]
    body = message.get_content()
    assert (host, port, timeout) == ("mailpit", 1025, 10)
    assert message["From"] == "security@example.test"
    assert message["To"] == "reset@example.com"
    assert "Expires at: 2026-09-18T12:30:00+00:00" in body
    assert "source=email&token=token+with+spaces%26symbols" in body
    assert "user_id" not in body
    assert "password_version" not in body


def test_smtp_password_reset_delivery_failures_are_sanitized(monkeypatch):
    class FailingSmtp:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("smtp private host detail raw-token")

    monkeypatch.setattr(password_reset_delivery.smtplib, "SMTP", FailingSmtp)
    service = SmtpPasswordResetDeliveryService(
        host="mailpit",
        port=1025,
        from_email="security@example.test",
        reset_url="http://localhost:3000/reset-password",
    )

    with pytest.raises(PasswordResetDeliveryError) as exc_info:
        service.deliver_password_reset(
            recipient_email="reset@example.com",
            token="raw-token",
            expires_at=EXPIRES_AT,
        )

    assert str(exc_info.value) == "Password reset delivery unavailable"
    assert "raw-token" not in str(exc_info.value)
    assert "smtp private" not in str(exc_info.value)


@pytest.mark.parametrize(
    "service",
    [
        SmtpPasswordResetDeliveryService(
            host=None,
            port=1025,
            from_email="security@example.test",
            reset_url="http://localhost:3000/reset-password",
        ),
        SmtpPasswordResetDeliveryService(
            host="mailpit",
            port=None,
            from_email="security@example.test",
            reset_url="http://localhost:3000/reset-password",
        ),
        SmtpPasswordResetDeliveryService(
            host="mailpit",
            port=1025,
            from_email=None,
            reset_url="http://localhost:3000/reset-password",
        ),
        SmtpPasswordResetDeliveryService(
            host="mailpit",
            port=1025,
            from_email="security@example.test",
            reset_url=None,
        ),
        DisabledPasswordResetDeliveryService(),
    ],
)
def test_password_reset_delivery_misconfiguration_is_sanitized(service):
    with pytest.raises(PasswordResetDeliveryError) as exc_info:
        service.deliver_password_reset(
            recipient_email="reset@example.com",
            token="raw-token",
            expires_at=EXPIRES_AT,
        )

    assert str(exc_info.value) == "Password reset delivery unavailable"
    assert "raw-token" not in str(exc_info.value)


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_settings_reject_missing_password_reset_url(app_env):
    with pytest.raises(ValidationError, match="PASSWORD_RESET_URL"):
        _deployed_settings(app_env=app_env, PASSWORD_RESET_URL=None)


@pytest.mark.parametrize(
    "url",
    [
        "http://app.example.com/reset-password",
        "http://localhost:3000/reset-password",
        "https://localhost/reset-password",
        "https://127.0.0.1/reset-password",
    ],
)
def test_deployed_settings_reject_insecure_password_reset_url(url):
    with pytest.raises(ValidationError, match="PASSWORD_RESET_URL"):
        _deployed_settings(PASSWORD_RESET_URL=url)


def test_deployed_settings_accept_secure_password_reset_url():
    settings = _deployed_settings(
        PASSWORD_RESET_URL="https://app.example.com/reset-password"
    )

    assert settings.PASSWORD_RESET_URL == "https://app.example.com/reset-password"


def _deployed_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "APP_ENV": "production",
        "DATABASE_URL": "postgresql://example",
        "API_KEY_PEPPER": "pepper",
        "DEV_COMPANY_ID": None,
        "LOGIN_RATE_LIMIT_BACKEND": "redis",
        "REDIS_URL": "redis://redis:6379/0",
        "TRUSTED_BROWSER_ORIGINS": ("https://app.example.com",),
        "INVITATION_DELIVERY_BACKEND": "smtp",
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": 587,
        "SMTP_FROM_EMAIL": "security@example.com",
        "INVITATION_ACCEPT_BASE_URL": "https://app.example.com/invitations/accept",
        "PASSWORD_RESET_URL": "https://app.example.com/reset-password",
    }
    values.update(overrides)
    return Settings(**values)
