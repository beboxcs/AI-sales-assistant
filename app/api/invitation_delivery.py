import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.config import settings
from app.domain.enums import MembershipRole


class InvitationDeliveryError(Exception):
    """Raised when an invitation cannot be delivered safely."""


class InvitationDeliveryService(Protocol):
    def deliver_invitation(
        self,
        *,
        recipient_email: str,
        token: str,
        expires_at: datetime,
        role: MembershipRole,
    ) -> None: ...


class SmtpInvitationDeliveryService:
    def __init__(
        self,
        *,
        host: str | None,
        port: int | None,
        from_email: str | None,
        accept_base_url: str | None,
    ) -> None:
        self._host = host
        self._port = port
        self._from_email = from_email
        self._accept_base_url = accept_base_url

    def deliver_invitation(
        self,
        *,
        recipient_email: str,
        token: str,
        expires_at: datetime,
        role: MembershipRole,
    ) -> None:
        if (
            not self._host
            or self._port is None
            or not self._from_email
            or not self._accept_base_url
        ):
            raise InvitationDeliveryError("Invitation delivery unavailable")
        try:
            message = EmailMessage()
            message["From"] = self._from_email
            message["To"] = recipient_email
            message["Subject"] = "You have been invited"
            message.set_content(
                "\n".join(
                    [
                        "You have been invited to join the Real Estate AI workspace.",
                        f"Role: {role.value}",
                        f"Expires at: {expires_at.isoformat()}",
                        f"Accept invitation: {_invitation_accept_url(self._accept_base_url, token)}",
                    ]
                )
            )
            with smtplib.SMTP(self._host, self._port, timeout=10) as smtp:
                smtp.send_message(message)
        except Exception as exc:
            raise InvitationDeliveryError("Invitation delivery unavailable") from exc


def get_invitation_delivery_service() -> InvitationDeliveryService:
    if settings.INVITATION_DELIVERY_BACKEND != "smtp":
        return DisabledInvitationDeliveryService()
    return SmtpInvitationDeliveryService(
        host=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        from_email=settings.SMTP_FROM_EMAIL,
        accept_base_url=settings.INVITATION_ACCEPT_BASE_URL,
    )


class DisabledInvitationDeliveryService:
    def deliver_invitation(
        self,
        *,
        recipient_email: str,
        token: str,
        expires_at: datetime,
        role: MembershipRole,
    ) -> None:
        raise InvitationDeliveryError("Invitation delivery unavailable")


def _invitation_accept_url(base_url: str, token: str) -> str:
    parsed = urlsplit(base_url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("token", token))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )
