"""
config/settings.py
------------------
All configuration is read from environment variables.
Never hardcode secrets, URIs, or model paths in application code.

Usage:
    from config.settings import settings
    print(settings.REDIS_URL)
"""

import os
from dataclasses import dataclass, field


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in os.getenv(name, default).split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class Settings:
    # ── Flask ──────────────────────────────────────────────────────────────
    FLASK_ENV: str = field(default_factory=lambda: os.getenv("FLASK_ENV", "production"))
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change-me-in-production"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", 5000)))
    FRONTEND_ORIGINS: tuple[str, ...] = field(default_factory=lambda: _csv_env(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,https://compass-two-iota.vercel.app,https://compaass.vercel.app,https://compass-seven-rho.vercel.app",
    ))

    # ── MongoDB ────────────────────────────────────────────────────────────
    MONGO_URI: str = field(default_factory=lambda: os.getenv(
        "MONGO_URI", "mongodb://localhost:27017"
    ))
    MONGO_DB_NAME: str = field(default_factory=lambda: os.getenv("MONGO_DB_NAME", "mental_health_chatbot"))
    MONGO_COLLECTION: str = field(default_factory=lambda: os.getenv("MONGO_COLLECTION", "conversations"))

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_URL: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    REDIS_MAX_CONNECTIONS: int = field(default_factory=lambda: int(os.getenv("REDIS_MAX_CONNECTIONS", 20)))
    SESSION_TTL_SECONDS: int = field(default_factory=lambda: int(os.getenv("SESSION_TTL_SECONDS", 1800)))  # 30 min
    PREDICTION_CACHE_TTL: int = field(default_factory=lambda: int(os.getenv("PREDICTION_CACHE_TTL", 60)))   # 1 min

    # ── Model ──────────────────────────────────────────────────────────────
    # Point to your fine-tuned DistilBERT directory (PyTorch) OR ONNX export
    MODEL_DIR: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "./distilbert_finetuned"))
    ONNX_MODEL_PATH: str = field(default_factory=lambda: os.getenv("ONNX_MODEL_PATH", "./onnx_model/model_quantized.onnx"))
    USE_ONNX: bool = field(default_factory=lambda: os.getenv("USE_ONNX", "true").lower() == "true")
    LABELS_PATH: str = field(default_factory=lambda: os.getenv("LABELS_PATH", "./label_classes.json"))

    # ── NLP ────────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", 0.50)))
    MAX_INPUT_LENGTH: int = field(default_factory=lambda: int(os.getenv("MAX_INPUT_LENGTH", 512)))
    MAX_RAW_CHARS: int = field(default_factory=lambda: int(os.getenv("MAX_RAW_CHARS", 1000)))
    SPACY_MODEL: str = field(default_factory=lambda: os.getenv("SPACY_MODEL", "en_core_web_sm"))

    # Language support
    ENABLE_MULTILINGUAL: bool = field(default_factory=lambda: os.getenv("ENABLE_MULTILINGUAL", "true").lower() == "true")
    SUPPORTED_LANGUAGES: tuple[str, ...] = field(default_factory=lambda: _csv_env("SUPPORTED_LANGUAGES", "en,yo,pcm"))
    LANGUAGE_TRANSLATION_PROVIDER: str = field(default_factory=lambda: os.getenv("LANGUAGE_TRANSLATION_PROVIDER", "groq"))
    LANGUAGE_TRANSLATION_TIMEOUT_SECONDS: float = field(default_factory=lambda: float(os.getenv("LANGUAGE_TRANSLATION_TIMEOUT_SECONDS", 8.0)))

    # LLM generation
    LLM_HISTORY_TURNS: int = field(default_factory=lambda: int(os.getenv("LLM_HISTORY_TURNS", 6)))
    LLM_TIMEOUT_SECONDS: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", 8.0)))
    LLM_MAX_TOKENS: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", 200)))

    # ── Rate Limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", 30)))
    RATE_LIMIT_WINDOW_SECONDS: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60)))

    # ── Therapist emergency alerts ───────────────────────────────────────────
    # When a hard crisis is detected, email an on-call therapist so a human can
    # follow up. Disabled unless ENABLE_THERAPIST_ALERTS=true AND SMTP + a
    # recipient are configured. Sending never blocks or breaks the user reply.
    ENABLE_THERAPIST_ALERTS: bool = field(default_factory=lambda: os.getenv("ENABLE_THERAPIST_ALERTS", "false").lower() == "true")
    THERAPIST_ALERT_EMAIL: str = field(default_factory=lambda: os.getenv("THERAPIST_ALERT_EMAIL", ""))
    ALERT_EMAIL_FROM: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL_FROM", ""))
    ALERT_APP_NAME: str = field(default_factory=lambda: os.getenv("ALERT_APP_NAME", "COMPASS"))
    # One alert per user per this many seconds (avoids spamming during an
    # ongoing crisis conversation). 0 disables de-duplication.
    ALERT_COOLDOWN_SECONDS: int = field(default_factory=lambda: int(os.getenv("ALERT_COOLDOWN_SECONDS", 900)))
    # SMTP transport. Works with Gmail, SendGrid SMTP relay, Mailgun, etc.
    SMTP_HOST: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    SMTP_PORT: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", 587)))
    SMTP_USERNAME: str = field(default_factory=lambda: os.getenv("SMTP_USERNAME", ""))
    SMTP_PASSWORD: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    SMTP_USE_TLS: bool = field(default_factory=lambda: os.getenv("SMTP_USE_TLS", "true").lower() == "true")
    SMTP_TIMEOUT_SECONDS: float = field(default_factory=lambda: float(os.getenv("SMTP_TIMEOUT_SECONDS", 10.0)))


# Singleton — import this everywhere
settings = Settings()
