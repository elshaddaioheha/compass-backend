"""
middleware/rate_limiter.py
--------------------------
Per-user rate limiting using Redis sliding window counters.

Algorithm: Fixed window counter (simple and effective for this scale).
    - Key: "rate:{user_id}"
    - Value: request count (integer)
    - TTL: RATE_LIMIT_WINDOW_SECONDS (resets the window)

If the user exceeds RATE_LIMIT_REQUESTS within the window, raise RateLimitError.

Why Redis and not Flask-Limiter:
    Redis-based limiting works across all Gunicorn workers (shared state).
    Flask-Limiter with in-memory storage only limits per-worker.

Usage:
    from middleware.rate_limiter import check_rate_limit, RateLimitError

    try:
        check_rate_limit(user_id)
    except RateLimitError as e:
        return jsonify({"error": str(e)}), 429
"""

from utils.redis_pool import get_redis
from utils.logger import log_error
from config.settings import settings


class RateLimitError(Exception):
    """Raised when a user exceeds their allowed request rate."""
    pass


def check_rate_limit(user_id: str) -> None:
    """
    Increment the request counter for this user and check against the limit.

    Args:
        user_id: A stable identifier for the user (session ID or user ID).

    Raises:
        RateLimitError: If the user has exceeded the rate limit.
    """
    redis = get_redis()
    key = f"rate:{user_id}"

    try:
        # Pipeline: INCR and EXPIRE in a single round-trip
        pipe = redis.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, settings.RATE_LIMIT_WINDOW_SECONDS)
        results = pipe.execute()

        count = results[0]  # result of INCR

        if count > settings.RATE_LIMIT_REQUESTS:
            raise RateLimitError(
                f"Too many messages. Please wait before sending again. "
                f"(Limit: {settings.RATE_LIMIT_REQUESTS} messages per "
                f"{settings.RATE_LIMIT_WINDOW_SECONDS} seconds)"
            )

    except RateLimitError:
        raise  # re-raise without wrapping
    except Exception as e:
        # If Redis is down, fail open (don't block the user)
        log_error(
            error=str(e),
            context="rate_limiter",
            user_id=user_id,
        )


def get_remaining_requests(user_id: str) -> int:
    """
    Returns how many requests the user has left in the current window.
    Returns RATE_LIMIT_REQUESTS if Redis is unavailable (fail open).
    """
    try:
        count = get_redis().get(f"rate:{user_id}")
        if count is None:
            return settings.RATE_LIMIT_REQUESTS
        used = int(count)
        return max(0, settings.RATE_LIMIT_REQUESTS - used)
    except Exception:
        return settings.RATE_LIMIT_REQUESTS
