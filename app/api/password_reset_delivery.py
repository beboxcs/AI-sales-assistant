import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.config import settings


class PasswordResetDeliveryError(Exception):
    """Raised when password reset instructions cannot be delivered safely."""


class PasswordResetDeliveryService(Protocol):
    def deliver_password_reset(
        self,
        *,
        recipient_email: str,
        token: str,
        expires_at: datetime,
    ) -> None: ...


class SmtpPasswordResetDeliveryService:
    def __init__(
        self,
        *,
        host: str | None,
        port: int | None,
        from_email: str | None,
        reset_url: str | None,
    ) -> None:
        self._host = host
        self._port = port
        self._from_email = from_email
        self._reset_url = reset_url

    def deliver_password_reset(
        self,
        *,
        recipient_email: str,
        token: str,
        expires_at: datetime,
    ) -> None:
        if (
            not self._host
            or self._port is None
            or not self._from_email
            or not self._reset_url
        ):
            raise PasswordResetDeliveryError("Password reset delivery unavailable")
        try:
            message = EmailMessage()
            message["From"] = self._from_email
            message["To"] = recipient_email
            message["Subject"] = "Reset your password"
            message.set_content(
                "\n".join(
                    [
                        "Use this link to reset your Real Estate AI password.",
                        f"Expires at: {expires_at.isoformat()}",
                        f"Reset password: {_password_reset_url(self._reset_url, token)}",
                    ]
                )
            )
            with smtplib.SMTP(self._host, self._port, timeout=10) as smtp:
                smtp.send_message(message)
        except Exception as exc:
            raise PasswordResetDeliveryError(
                "Password reset delivery unavailable"
            ) from exc


def get_password_reset_delivery_service() -> PasswordResetDeliveryService:
    if settings.INVITATION_DELIVERY_BACKEND != "smtp":
        return DisabledPasswordResetDeliveryService()
    return SmtpPasswordResetDeliveryService(
        host=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        from_email=settings.SMTP_FROM_EMAIL,
        reset_url=settings.PASSWORD_RESET_URL,
    )


class DisabledPasswordResetDeliveryService:
    def deliver_password_reset(
        self,
        *,
        recipient_email: str,
        token: str,
        expires_at: datetime,
    ) -> None:
        raise PasswordResetDeliveryError("Password reset delivery unavailable")


def _password_reset_url(base_url: str, token: str) -> str:
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
