from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.api import invitation_delivery
from app.api.invitation_delivery import (
    InvitationDeliveryError,
    SmtpInvitationDeliveryService,
)
from app.config import Settings
from app.domain.enums import MembershipRole


def test_smtp_delivery_sends_invitation_link_with_encoded_token(monkeypatch):
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

    monkeypatch.setattr(invitation_delivery.smtplib, "SMTP", FakeSmtp)
    service = SmtpInvitationDeliveryService(
        host="mailpit",
        port=1025,
        from_email="invites@example.test",
        accept_base_url="http://localhost:3000/invitations/accept?source=email",
    )

    service.deliver_invitation(
        recipient_email="sales@example.com",
        token="token with spaces&symbols",
        expires_at=datetime(2026, 9, 18, 12, 30, tzinfo=timezone.utc),
        role=MembershipRole.SALESPERSON,
    )

    assert len(sent_messages) == 1
    host, port, timeout, message = sent_messages[0]
    body = message.get_content()
    assert (host, port, timeout) == ("mailpit", 1025, 10)
    assert message["From"] == "invites@example.test"
    assert message["To"] == "sales@example.com"
    assert "Role: salesperson" in body
    assert "source=email&token=token+with+spaces%26symbols" in body
    assert "company_id" not in body


def test_smtp_delivery_failures_are_sanitized(monkeypatch):
    class FailingSmtp:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("smtp private host detail raw-token")

    monkeypatch.setattr(invitation_delivery.smtplib, "SMTP", FailingSmtp)
    service = SmtpInvitationDeliveryService(
        host="mailpit",
        port=1025,
        from_email="invites@example.test",
        accept_base_url="http://localhost:3000/invitations/accept",
    )

    with pytest.raises(InvitationDeliveryError) as exc_info:
        service.deliver_invitation(
            recipient_email="sales@example.com",
            token="raw-token",
            expires_at=datetime.now(timezone.utc),
            role=MembershipRole.COMPANY_ADMIN,
        )

    assert str(exc_info.value) == "Invitation delivery unavailable"
    assert "raw-token" not in str(exc_info.value)
    assert "smtp private" not in str(exc_info.value)


def test_deployed_settings_require_configured_smtp_delivery():
    with pytest.raises(ValidationError, match="INVITATION_DELIVERY_BACKEND"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            DATABASE_URL="postgresql://example",
            API_KEY_PEPPER="pepper",
            DEV_COMPANY_ID=None,
            INVITATION_DELIVERY_BACKEND=None,
            SMTP_HOST=None,
            SMTP_PORT=None,
            SMTP_FROM_EMAIL=None,
            INVITATION_ACCEPT_BASE_URL=None,
        )


def test_deployed_settings_reject_mailpit_delivery_host():
    with pytest.raises(ValidationError, match="development-only"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            DATABASE_URL="postgresql://example",
            API_KEY_PEPPER="pepper",
            DEV_COMPANY_ID=None,
            INVITATION_DELIVERY_BACKEND="smtp",
            SMTP_HOST="mailpit",
            SMTP_PORT=1025,
            SMTP_FROM_EMAIL="invites@example.com",
            INVITATION_ACCEPT_BASE_URL="https://app.example.com/invitations/accept",
        )
