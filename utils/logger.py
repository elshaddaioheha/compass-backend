"""
utils/logger.py
---------------
Structured JSON logging for the NLP layer.

Every log record is a JSON object, not a plain string.
This makes logs machine-parseable by tools like MongoDB, Datadog, or grep.

Fields logged on every prediction:
    - timestamp (ISO 8601)
    - event type
    - user_id
    - emotion predicted
    - confidence score
    - latency_ms
    - input_length (character count, NOT the raw text for privacy)

Usage:
    from utils.logger import log_prediction, log_error, log_request

    log_prediction(user_id="abc", emotion="anxiety", confidence=0.91, latency_ms=43.2)
    log_error(user_id="abc", error="Redis timeout", context="session_read")
    log_request(user_id="abc", route="/send", status=200, latency_ms=55.1)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_record(event: str, user_id: Optional[str] = None) -> dict:
    return {
        "timestamp": _now(),
        "event": event,
        "user_id": user_id or "anonymous",
    }


# ── Logger setup ──────────────────────────────────────────────────────────────
# One logger, writes to stdout so Gunicorn / Docker can capture it
_logger = logging.getLogger("nlp_layer")
_logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))   # raw JSON, no extra wrapping
_logger.addHandler(_handler)
_logger.propagate = False


def _emit(record: dict) -> None:
    _logger.info(json.dumps(record, ensure_ascii=False))


# ── Public API ────────────────────────────────────────────────────────────────

def log_prediction(
    user_id: str,
    emotion: str,
    confidence: float,
    latency_ms: float,
    input_length: int,
    cache_hit: bool = False,
    low_confidence: bool = False,
) -> None:
    """Log a completed emotion prediction."""
    record = _base_record("prediction", user_id)
    record.update({
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
        "input_length": input_length,
        "cache_hit": cache_hit,
        "low_confidence": low_confidence,
    })
    _emit(record)


def log_request(
    user_id: str,
    route: str,
    method: str,
    status: int,
    latency_ms: float,
) -> None:
    """Log an HTTP request to the NLP API."""
    record = _base_record("http_request", user_id)
    record.update({
        "route": route,
        "method": method,
        "status": status,
        "latency_ms": round(latency_ms, 2),
    })
    _emit(record)


def log_error(
    error: str,
    context: str,
    user_id: Optional[str] = None,
) -> None:
    """Log an error with context so it's easy to trace."""
    record = _base_record("error", user_id)
    record.update({
        "error": error,
        "context": context,
    })
    _logger.error(json.dumps(record, ensure_ascii=False))


def log_crisis(
    user_id: str,
    emotion: str,
    confidence: float,
    triggered_by: str,
) -> None:
    """Log a crisis detection event — critical priority."""
    record = _base_record("crisis_detected", user_id)
    record.update({
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "triggered_by": triggered_by,   # e.g. "keyword" or "model"
    })
    _logger.critical(json.dumps(record, ensure_ascii=False))


def log_session(
    user_id: str,
    action: str,  # "created", "resumed", "expired"
    cbt_active: bool = False,
) -> None:
    """Log session lifecycle events."""
    record = _base_record("session", user_id)
    record.update({"action": action, "cbt_active": cbt_active})
    _emit(record)
