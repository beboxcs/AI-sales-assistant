import pytest
from pydantic import ValidationError

from app.config import Settings


def deployed_settings_kwargs() -> dict[str, object]:
    return {
        "DATABASE_URL": "postgresql://example",
        "API_KEY_PEPPER": "pepper",
        "DEV_COMPANY_ID": None,
        "TRUSTED_BROWSER_ORIGINS": ("https://app.example.com",),
        "INVITATION_DELIVERY_BACKEND": "smtp",
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": 587,
        "SMTP_FROM_EMAIL": "invites@example.com",
        "INVITATION_ACCEPT_BASE_URL": "https://app.example.com/invitations/accept",
        "PASSWORD_RESET_URL": "https://app.example.com/reset-password",
    }


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
    assert settings.LOGIN_RATE_LIMIT_BACKEND is None


def test_login_rate_limit_backend_loads_from_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("LOGIN_RATE_LIMIT_BACKEND", "memory")

    settings = Settings(_env_file=None)

    assert settings.LOGIN_RATE_LIMIT_BACKEND == "memory"


@pytest.mark.parametrize("app_env", ["local", "test"])
def test_local_and_test_allow_missing_login_rate_limit_backend(app_env):
    settings = Settings(
        _env_file=None,
        APP_ENV=app_env,
        DATABASE_URL="postgresql://example",
        LOGIN_RATE_LIMIT_BACKEND=None,
    )

    assert settings.LOGIN_RATE_LIMIT_BACKEND is None


@pytest.mark.parametrize("app_env", ["local", "test"])
def test_local_and_test_allow_memory_login_rate_limit_backend(app_env):
    settings = Settings(
        _env_file=None,
        APP_ENV=app_env,
        DATABASE_URL="postgresql://example",
        LOGIN_RATE_LIMIT_BACKEND="memory",
    )

    assert settings.LOGIN_RATE_LIMIT_BACKEND == "memory"


@pytest.mark.parametrize("app_env", ["local", "test"])
def test_local_and_test_allow_redis_login_rate_limit_backend(app_env):
    settings = Settings(
        _env_file=None,
        APP_ENV=app_env,
        DATABASE_URL="postgresql://example",
        LOGIN_RATE_LIMIT_BACKEND="redis",
        REDIS_URL="redis://localhost:6379/0",
    )

    assert settings.LOGIN_RATE_LIMIT_BACKEND == "redis"
    assert settings.REDIS_URL == "redis://localhost:6379/0"


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_config_allows_redis_login_rate_limit_backend(app_env):
    settings = Settings(
        _env_file=None,
        APP_ENV=app_env,
        LOGIN_RATE_LIMIT_BACKEND="redis",
        REDIS_URL="redis://redis:6379/0",
        **deployed_settings_kwargs(),
    )

    assert settings.LOGIN_RATE_LIMIT_BACKEND == "redis"


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_config_rejects_memory_login_rate_limit_backend(app_env):
    with pytest.raises(ValidationError, match="LOGIN_RATE_LIMIT_BACKEND=redis"):
        Settings(
            _env_file=None,
            APP_ENV=app_env,
            LOGIN_RATE_LIMIT_BACKEND="memory",
            **deployed_settings_kwargs(),
        )


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_config_rejects_missing_login_rate_limit_backend(app_env):
    with pytest.raises(ValidationError, match="LOGIN_RATE_LIMIT_BACKEND"):
        Settings(
            _env_file=None,
            APP_ENV=app_env,
            LOGIN_RATE_LIMIT_BACKEND=None,
            **deployed_settings_kwargs(),
        )


def test_login_rate_limit_backend_rejects_unknown_backend():
    with pytest.raises(ValidationError, match="LOGIN_RATE_LIMIT_BACKEND"):
        Settings(
            _env_file=None,
            APP_ENV="local",
            DATABASE_URL="postgresql://example",
            LOGIN_RATE_LIMIT_BACKEND="sqlite",
        )


def test_redis_login_rate_limit_backend_requires_redis_url():
    with pytest.raises(ValidationError, match="REDIS_URL"):
        Settings(
            _env_file=None,
            APP_ENV="local",
            DATABASE_URL="postgresql://example",
            LOGIN_RATE_LIMIT_BACKEND="redis",
            REDIS_URL=None,
        )


def test_redis_login_rate_limit_backend_rejects_malformed_redis_url():
    with pytest.raises(ValidationError, match="REDIS_URL"):
        Settings(
            _env_file=None,
            APP_ENV="local",
            DATABASE_URL="postgresql://example",
            LOGIN_RATE_LIMIT_BACKEND="redis",
            REDIS_URL="redis://localhost:bad/0",
        )


def test_redis_login_rate_limit_backend_rejects_unsupported_redis_url_scheme():
    with pytest.raises(ValidationError, match="REDIS_URL"):
        Settings(
            _env_file=None,
            APP_ENV="local",
            DATABASE_URL="postgresql://example",
            LOGIN_RATE_LIMIT_BACKEND="redis",
            REDIS_URL="http://localhost:6379/0",
        )


def test_redis_login_rate_limit_backend_accepts_rediss_url():
    settings = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        LOGIN_RATE_LIMIT_BACKEND="redis",
        REDIS_URL="rediss://redis.example.com:6379/0",
    )

    assert settings.REDIS_URL == "rediss://redis.example.com:6379/0"


@pytest.mark.parametrize(
    "query_parameter",
    [
        "decode_responses=true",
        "socket_connect_timeout=30",
        "socket_timeout=30",
    ],
)
def test_redis_login_rate_limit_backend_rejects_frozen_client_option_query_parameters(
    query_parameter,
):
    with pytest.raises(ValidationError, match="client option query parameters"):
        Settings(
            _env_file=None,
            APP_ENV="local",
            DATABASE_URL="postgresql://example",
            LOGIN_RATE_LIMIT_BACKEND="redis",
            REDIS_URL=f"redis://localhost:6379/0?{query_parameter}",
        )


def test_redis_url_validation_does_not_connect_to_redis():
    settings = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        LOGIN_RATE_LIMIT_BACKEND="redis",
        REDIS_URL="redis://127.0.0.1:1/0",
    )

    assert settings.REDIS_URL == "redis://127.0.0.1:1/0"


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
            LOGIN_RATE_LIMIT_BACKEND="redis",
            REDIS_URL="redis://redis:6379/0",
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
            LOGIN_RATE_LIMIT_BACKEND="redis",
            REDIS_URL="redis://redis:6379/0",
        )


def test_production_config_rejects_wildcard_trusted_browser_origin():
    with pytest.raises(ValidationError, match="exact origins"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            DATABASE_URL="postgresql://example",
            API_KEY_PEPPER="pepper",
            DEV_COMPANY_ID=None,
            LOGIN_RATE_LIMIT_BACKEND="redis",
            REDIS_URL="redis://redis:6379/0",
            TRUSTED_BROWSER_ORIGINS=("*",),
        )
