from redis import Redis

from app.config import Settings, settings


def create_redis_client(config: Settings = settings) -> Redis:
    if config.REDIS_URL is None:
        raise ValueError("REDIS_URL is required to create a Redis client")

    return Redis.from_url(
        config.REDIS_URL,
        decode_responses=False,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
