import pytest
from pydantic import ValidationError

from app.config import Settings


def test_local_config_allows_explicit_dev_company_id_without_pepper():
    settings = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        API_KEY_PEPPER=None,
        DEV_COMPANY_ID=1,
    )

    assert settings.DEV_COMPANY_ID == 1
    assert settings.API_KEY_PEPPER is None


def test_test_config_allows_dependency_override_style_configuration():
    settings = Settings(
        _env_file=None,
        APP_ENV="test",
        DATABASE_URL="postgresql://example",
        API_KEY_PEPPER=None,
        DEV_COMPANY_ID=42,
    )

    assert settings.APP_ENV == "test"
    assert settings.DEV_COMPANY_ID == 42


def test_trusted_browser_origins_are_normalized():
    settings = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        TRUSTED_BROWSER_ORIGINS=(
            " HTTPS://App.Example.com/ ",
            "http://localhost:3000",
        ),
    )

    assert settings.TRUSTED_BROWSER_ORIGINS == (
        "https://app.example.com",
        "http://localhost:3000",
    )


def test_trusted_browser_origin_default_ports_normalize_away():
    settings = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        TRUSTED_BROWSER_ORIGINS=(
            "https://app.example.com:443",
            "http://app.example.com:80",
            "https://app.example.com:8443",
            "http://localhost:3000",
        ),
    )

    assert settings.TRUSTED_BROWSER_ORIGINS == (
        "https://app.example.com",
        "http://app.example.com",
        "https://app.example.com:8443",
        "http://localhost:3000",
    )


def test_trusted_browser_origins_load_from_json_array_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv(
        "TRUSTED_BROWSER_ORIGINS",
        '["https://app.example.com:443", "http://localhost:3000"]',
    )

    settings = Settings(_env_file=None)

    assert settings.TRUSTED_BROWSER_ORIGINS == (
        "https://app.example.com",
        "http://localhost:3000",
    )


def test_missing_app_env_is_configuration_error(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)

    with pytest.raises(ValidationError, match="APP_ENV"):
        Settings(
            _env_file=None,
            DATABASE_URL="postgresql://example",
            API_KEY_PEPPER=None,
            DEV_COMPANY_ID=1,
        )


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_config_rejects_dev_company_id(app_env):
    with pytest.raises(ValidationError, match="DEV_COMPANY_ID"):
        Settings(
            _env_file=None,
            APP_ENV=app_env,
            DATABASE_URL="postgresql://example",
            API_KEY_PEPPER="pepper",
            DEV_COMPANY_ID=1,
        )


@pytest.mark.parametrize("app_env", ["staging", "production"])
@pytest.mark.parametrize("pepper", [None, "", "   "])
def test_deployed_config_requires_non_blank_api_key_pepper(app_env, pepper):
    with pytest.raises(ValidationError, match="API_KEY_PEPPER"):
        Settings(
            _env_file=None,
            APP_ENV=app_env,
            DATABASE_URL="postgresql://example",
            API_KEY_PEPPER=pepper,
            DEV_COMPANY_ID=None,
        )


def test_production_config_rejects_wildcard_trusted_browser_origin():
    with pytest.raises(ValidationError, match="exact origins"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            DATABASE_URL="postgresql://example",
            API_KEY_PEPPER="pepper",
            DEV_COMPANY_ID=None,
            TRUSTED_BROWSER_ORIGINS=("*",),
        )
