from typing import Literal
from urllib.parse import urlsplit

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_browser_origin(origin: str) -> str:
    if not isinstance(origin, str):
        raise ValueError("TRUSTED_BROWSER_ORIGINS must contain strings")

    candidate = origin.strip()
    if candidate == "*" or "*" in candidate:
        raise ValueError("TRUSTED_BROWSER_ORIGINS must contain exact origins")

    parsed = urlsplit(candidate)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("TRUSTED_BROWSER_ORIGINS must use http or https")
    if not parsed.hostname:
        raise ValueError("TRUSTED_BROWSER_ORIGINS must include a host")
    if parsed.username or parsed.password:
        raise ValueError("TRUSTED_BROWSER_ORIGINS must not include credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("TRUSTED_BROWSER_ORIGINS must be origins only")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("TRUSTED_BROWSER_ORIGINS has an invalid port") from exc

    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    normalized = f"{scheme}://{host}"
    if port is not None and port != _default_origin_port(scheme):
        normalized = f"{normalized}:{port}"
    return normalized


def _default_origin_port(scheme: str) -> int:
    if scheme == "http":
        return 80
    return 443


class Settings(BaseSettings):
    APP_ENV: Literal["local", "test", "staging", "production"]
    DATABASE_URL: str
    API_KEY_PEPPER: str | None = None
    DEV_COMPANY_ID: int | None = None
    INVENTORY_INGEST_MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    TRUSTED_BROWSER_ORIGINS: tuple[str, ...] = ()

    @field_validator("TRUSTED_BROWSER_ORIGINS", mode="before")
    @classmethod
    def normalize_trusted_browser_origins(cls, value) -> tuple[str, ...]:
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            origins = (value,)
        else:
            origins = tuple(value)
        return tuple(normalize_browser_origin(origin) for origin in origins)

    @model_validator(mode="after")
    def validate_auth_configuration(self) -> "Settings":
        if self.APP_ENV in {"staging", "production"}:
            if self.DEV_COMPANY_ID is not None:
                raise ValueError(
                    "DEV_COMPANY_ID must not be configured outside local/test"
                )
            if self.API_KEY_PEPPER is None or not self.API_KEY_PEPPER.strip():
                raise ValueError(
                    "API_KEY_PEPPER is required outside local/test"
                )
            if any(origin == "*" for origin in self.TRUSTED_BROWSER_ORIGINS):
                raise ValueError(
                    "TRUSTED_BROWSER_ORIGINS must not contain wildcard origins"
                )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
