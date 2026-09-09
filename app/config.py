from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: Literal["local", "test", "staging", "production"]
    DATABASE_URL: str
    API_KEY_PEPPER: str | None = None
    DEV_COMPANY_ID: int | None = None
    INVENTORY_INGEST_MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024

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
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
