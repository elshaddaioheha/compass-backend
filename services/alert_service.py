"""
services/alert_service.py
-------------------------
Therapist emergency alerting.

When the pipeline detects a hard crisis (suicidal / self-harm signals), this
module emails an on-call therapist so a human can follow up promptly. The user
still receives the fixed crisis template with hotlines either way — this layer
escalates to a real person in parallel.

Design rules (all safety-critical):
    - FAIL-SAFE: alerting must NEVER block or break the user's reply. The SMTP
      send runs in a background thread and every error is logged, not raised.
    - DE-DUPLICATED: at most one alert per user per ALERT_COOLDOWN_SECONDS, so an
      ongoing crisis conversation doesn't flood the therapist. Backed by Redis
      when available, with an in-process fallback.
    - OPT-IN: does nothing unless ENABLE_THERAPIST_ALERTS=true and SMTP + a
      recipient are configured. With it off, behavior is unchanged.

Usage (from the pipeline's crisis branch):
    from services.alert_service import maybe_alert_therapist

    maybe_alert_therapist(
        user_id=user_id,
        message="I want to end my life",
        emotion="sadness",
        confidence=0.88,
        triggered_by="keyword:end my life",
        language="en",
    )
"""

import smtplib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Optional

from config.settings import settings
from utils.logger import log_error, log_session

# In-process cooldown fallback (used only when Redis is unavailable).
_recent_alerts: dict[str, float] = {}
_recent_lock = threading.Lock()


@dataclass(frozen=True)
class AlertPayload:
    user_id: str
    message: str
    emotion: str
    confidence: float
    triggered_by: str
    language: str = "en"
    translated_message: Optional[str] = None


# ── Gating ────────────────────────────────────────────────────────────────────

def _alerts_configured() -> bool:
    """True only if alerting is enabled and the minimum config is present."""
    return bool(
        settings.ENABLE_THERAPIST_ALERTS
        and settings.THERAPIST_ALERT_EMAIL
        and settings.SMTP_HOST
    )


def _should_send(user_id: str) -> bool:
    """
    Cooldown check — return True at most once per user per cooldown window.

    Prefers a Redis key with NX+TTL (shared across workers). Falls back to an
    in-process timestamp map if Redis is unavailable.
    """
    cooldown = settings.ALERT_COOLDOWN_SECONDS
    if cooldown <= 0:
        return True

    key = f"alert_cooldown:{user_id}"
    try:
        from utils.redis_pool import get_redis
        # set(..., nx=True) succeeds only if the key does not already exist.
        was_set = get_redis().set(key, "1", nx=True, ex=cooldown)
        return bool(was_set)
    except Exception:
        # Redis down — use the in-process fallback so we still de-duplicate.
        now = time.monotonic()
        with _recent_lock:
            last = _recent_alerts.get(user_id, 0.0)
            if now - last < cooldown:
                return False
            _recent_alerts[user_id] = now
            return True


# ── Message building (pure — easy to test) ─────────────────────────────────────

def build_alert_message(payload: AlertPayload) -> tuple[str, str]:
    """Return (subject, plaintext_body) for a crisis alert email."""
    app = settings.ALERT_APP_NAME
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    subject = f"[{app}] \U0001F6A8 Crisis alert — session {payload.user_id}"

    lines = [
        f"{app} detected a possible mental-health emergency and is escalating to you.",
        "",
        "The user has been shown crisis resources and the Nigerian hotline numbers,",
        "but please reach out and follow up as soon as possible.",
        "",
        "── Details ─────────────────────────────────────────────",
        f"Time:            {when}",
        f"Session ID:      {payload.user_id}",
        f"Detected emotion: {payload.emotion} (confidence {payload.confidence:.0%})",
        f"Language:        {payload.language}",
        f"Triggered by:    {payload.triggered_by}",
        "",
        "User message:",
        f"  {payload.message}",
    ]
    if payload.translated_message and payload.translated_message != payload.message:
        lines += ["", "English translation:", f"  {payload.translated_message}"]
    lines += [
        "",
        "────────────────────────────────────────────────────────",
        f"This is an automated alert from {app}. Do not reply to this email.",
    ]
    return subject, "\n".join(lines)


# ── SMTP transport ──────────────────────────────────────────────────────────────

def _send_email(subject: str, body: str) -> None:
    """Send a plaintext email via SMTP. Raises on failure (caller logs it)."""
    msg = EmailMessage()
    msg["From"] = settings.ALERT_EMAIL_FROM or settings.SMTP_USERNAME
    msg["To"] = settings.THERAPIST_ALERT_EMAIL
    msg["Subject"] = subject
    msg.set_content(body)

    timeout = settings.SMTP_TIMEOUT_SECONDS
    if settings.SMTP_PORT == 465:
        # Implicit TLS
        with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=timeout) as server:
            if settings.SMTP_USERNAME:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)
    else:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=timeout) as server:
            if settings.SMTP_USE_TLS:
                server.starttls()
            if settings.SMTP_USERNAME:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)


def _dispatch(payload: AlertPayload) -> None:
    """Build + send the alert. Swallows all errors (runs in a worker thread)."""
    try:
        subject, body = build_alert_message(payload)
        _send_email(subject, body)
        log_session(user_id=payload.user_id, action="therapist_alert_sent")
    except Exception as e:
        log_error(error=str(e), context="therapist_alert", user_id=payload.user_id)


# ── Public API ──────────────────────────────────────────────────────────────────

def maybe_alert_therapist(
    user_id: str,
    message: str,
    emotion: str,
    confidence: float,
    triggered_by: str,
    language: str = "en",
    translated_message: Optional[str] = None,
) -> bool:
    """
    Send a therapist alert for a crisis turn, if enabled and not on cooldown.

    Returns True if an alert was dispatched, False if it was skipped (disabled,
    unconfigured, or within the cooldown window). Never raises.
    """
    try:
        if not _alerts_configured():
            return False
        if not _should_send(user_id):
            return False

        payload = AlertPayload(
            user_id=user_id,
            message=message,
            emotion=emotion,
            confidence=confidence,
            triggered_by=triggered_by,
            language=language,
            translated_message=translated_message,
        )
        # Background thread — the user's crisis reply must not wait on SMTP.
        threading.Thread(target=_dispatch, args=(payload,), daemon=True).start()
        return True
    except Exception as e:
        log_error(error=str(e), context="therapist_alert_gate", user_id=user_id)
        return False
