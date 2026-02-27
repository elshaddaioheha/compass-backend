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


@dataclass(frozen=True)
class Settings:
    # ── Flask ──────────────────────────────────────────────────────────────
    FLASK_ENV: str = field(default_factory=lambda: os.getenv("FLASK_ENV", "production"))
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change-me-in-production"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", 5000)))

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
    CONFIDENCE_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", 0.55)))
    MAX_INPUT_LENGTH: int = field(default_factory=lambda: int(os.getenv("MAX_INPUT_LENGTH", 512)))
    MAX_RAW_CHARS: int = field(default_factory=lambda: int(os.getenv("MAX_RAW_CHARS", 1000)))
    SPACY_MODEL: str = field(default_factory=lambda: os.getenv("SPACY_MODEL", "en_core_web_sm"))

    # ── Rate Limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", 30)))
    RATE_LIMIT_WINDOW_SECONDS: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60)))


# Singleton — import this everywhere
settings = Settings()
