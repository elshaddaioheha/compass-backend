"""
utils/redis_pool.py
-------------------
Single shared Redis connection pool for the entire application.

Why this matters:
    Without a pool, every request opens and closes a TCP connection to Redis.
    At 30 req/min per user with 4 Gunicorn workers, that's hundreds of
    wasted connections per minute. A pool keeps connections alive and reuses them.

Usage:
    from utils.redis_pool import get_redis
    r = get_redis()
    r.set("key", "value")
"""

import redis
from config.settings import settings


# Module-level pool — created once when this module is first imported
# Short timeouts (0.5s) ensure that when Redis is unavailable, the app
# fails-open quickly rather than making each request wait 2+ seconds.
_pool = redis.ConnectionPool.from_url(
    settings.REDIS_URL,
    max_connections=settings.REDIS_MAX_CONNECTIONS,
    decode_responses=True,   # return strings, not bytes
    socket_connect_timeout=0.5,   # fail fast so fail-open is snappy
    socket_timeout=0.5,
    retry_on_timeout=False,       # don't retry — just fail-open immediately
)


def get_redis() -> redis.Redis:
    """Return a Redis client backed by the shared connection pool."""
    return redis.Redis(connection_pool=_pool)


def ping() -> bool:
    """Health check — returns True if Redis is reachable."""
    try:
        return get_redis().ping()
    except redis.RedisError:
        return False
