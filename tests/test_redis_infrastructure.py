from unittest.mock import patch

from app.config import Settings
from app.infrastructure.redis import create_redis_client


def test_create_redis_client_uses_configured_timeouts_and_binary_responses():
    config = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        LOGIN_RATE_LIMIT_BACKEND="redis",
        REDIS_URL="redis://localhost:6379/0",
    )
    client = object()

    with patch("app.infrastructure.redis.Redis.from_url", return_value=client) as from_url:
        result = create_redis_client(config)

    assert result is client
    from_url.assert_called_once_with(
        "redis://localhost:6379/0",
        decode_responses=False,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def test_create_redis_client_does_not_ping_or_connect_eagerly():
    config = Settings(
        _env_file=None,
        APP_ENV="local",
        DATABASE_URL="postgresql://example",
        LOGIN_RATE_LIMIT_BACKEND="redis",
        REDIS_URL="redis://localhost:6379/0",
    )

    with patch("app.infrastructure.redis.Redis.from_url") as from_url:
        create_redis_client(config)

    assert from_url.return_value.ping.call_count == 0
    assert from_url.return_value.connection_pool.get_connection.call_count == 0
