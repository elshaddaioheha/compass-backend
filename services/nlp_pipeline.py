"""
services/nlp_pipeline.py
------------------------
Orchestrates the full NLP pipeline for one user message:

    Raw input
        ↓
    InputValidator       (reject bad input early)
        ↓
    RateLimiter          (reject if user is sending too fast)
        ↓
    Preprocessor         (clean text)
        ↓
    EmotionClassifier    (DistilBERT / ONNX → emotion + confidence)
        ↓
    DialogueManager      (update session state → generate reply)
        ↓
    MongoLogger          (persist conversation record)
        ↓
    Structured log       (latency, emotion, confidence)
        ↓
    Response dict

This is the single function Flask routes should call.
It handles its own error catching so Flask routes stay simple.

Usage:
    from services.nlp_pipeline import process_message

    result = process_message(
        raw_text="I can't sleep and I'm really anxious",
        user_id="session_abc123",
    )
    # result["reply"] → chatbot reply string
    # result["emotion"] → detected emotion
    # result["confidence"] → float
    # result["error"] → None or error message string
"""

import os
import time
from collections import defaultdict
from typing import Optional
from datetime import datetime, timezone

from middleware.input_validator import validate_input, InputValidationError
from middleware.rate_limiter import check_rate_limit, RateLimitError
from services.preprocessor import preprocess
from models.emotion_classifier import classifier
from services.dialogue_manager import DialogueManager
from services.language_service import prepare_language_context, translate_reply
from services.llm_service import generate_reply, LLMUnavailable
from utils.logger import log_prediction, log_error, log_request


# Single DialogueManager instance — stateless, safe to share
_dm = DialogueManager()

# In-process conversation history per user (last 4 turns = 2 exchanges)
_history: dict[str, list] = defaultdict(list)
_MAX_HISTORY_TURNS = 4

# ── Startup: confirm LLM status ───────────────────────────────────────────────
_groq_key = os.getenv("GROQ_API_KEY", "")
if _groq_key:
    print(f"[pipeline] GROQ_API_KEY loaded (key starts with: {_groq_key[:8]}...)")
else:
    print("[pipeline] GROQ_API_KEY not set - will use template replies.")
    print("           Add GROQ_API_KEY=gsk_... to your .env file.")


def _log_to_mongo(
    db,                   # pymongo database object (passed from app.py)
    user_id: str,
    raw_message: str,
    processing_message: str,
    emotion: str,
    confidence: float,
    reply: str,
    latency_ms: float,
    language_context=None,
) -> None:
    """
    Persist conversation record to MongoDB.
    Non-blocking: errors are logged but do not affect the response.
    """
    try:
        record = {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_message": raw_message,
            "processing_message": processing_message,
            "detected_emotion": emotion,
            "confidence": round(confidence, 4),
            "bot_reply": reply,
            "latency_ms": round(latency_ms, 2),
        }
        if language_context is not None:
            record["language"] = {
                "detected": language_context.detected_language,
                "reply": language_context.reply_language,
                "provider": language_context.provider,
                "input_translation_applied": language_context.input_translation_applied,
                "input_translation_error": language_context.input_translation_error,
                "detection_confidence": round(language_context.detection_confidence, 4),
            }
        db[_db_collection].insert_one(record)
    except Exception as e:
        log_error(error=str(e), context="mongo_log", user_id=user_id)


# Collection name — read once
try:
    from config.settings import settings as _settings
    _db_collection = _settings.MONGO_COLLECTION
except Exception:
    _db_collection = "conversations"


def process_message(
    raw_text: str,
    user_id: str,
    mongo_db=None,     # optional: pass pymongo db for persistence
    requested_language: Optional[str] = None,
    requested_reply_language: Optional[str] = None,
) -> dict:
    """
    Run the full NLP pipeline for one user message.

    Args:
        raw_text:   Raw string from the HTTP request body.
        user_id:    Stable session or user identifier.
        mongo_db:   Optional pymongo Database object for logging.

    Returns:
        dict with keys:
            reply       (str)   — chatbot reply
            emotion     (str)   — detected emotion label
            confidence  (float) — model confidence 0–1
            latency_ms  (float) — total pipeline latency
            cache_hit   (bool)  — True if prediction was served from cache
            error       (str|None) — error message if validation/rate failed
            status_code (int)   — HTTP status to return (200, 400, 429, 500)
    """
    t_pipeline_start = time.perf_counter()

    # ── 1. Input validation ────────────────────────────────────────────────
    try:
        clean_text = validate_input(raw_text)
    except InputValidationError as e:
        return {
            "reply": str(e),
            "emotion": None,
            "confidence": None,
            "latency_ms": 0.0,
            "cache_hit": False,
            "error": str(e),
            "status_code": 400,
        }

    # ── 2. Rate limiting ───────────────────────────────────────────────────
    try:
        check_rate_limit(user_id)
    except RateLimitError as e:
        return {
            "reply": str(e),
            "emotion": None,
            "confidence": None,
            "latency_ms": 0.0,
            "cache_hit": False,
            "error": str(e),
            "status_code": 429,
        }

    # ── 3. Language detection / translation-in ─────────────────────────────
    language_context = prepare_language_context(
        clean_text,
        requested_language=requested_language,
        requested_reply_language=requested_reply_language,
    )
    processing_text = language_context.processing_text

    # ── 4. Preprocessing ───────────────────────────────────────────────────
    preprocessed = preprocess(processing_text)

    # ── 5. Emotion classification ──────────────────────────────────────────
    prediction = classifier.predict(preprocessed)

    log_prediction(
        user_id=user_id,
        emotion=prediction.emotion,
        confidence=prediction.confidence,
        latency_ms=prediction.latency_ms,
        input_length=len(preprocessed),
        cache_hit=prediction.cache_hit,
        low_confidence=prediction.low_confidence,
    )

    # ── 6. Dialogue state update ───────────────────────────────────────────
    updated_state = _dm.update_state(
        user_id=user_id,
        emotion=prediction.emotion,
        confidence=prediction.confidence,
        message=preprocessed,
    )
    if language_context.crisis_signal:
        updated_state["crisis_flag"] = True
        updated_state["crisis_triggered_by"] = language_context.crisis_signal_source

    # ── 7. Generate reply ──────────────────────────────────────────────────
    # Crisis is always handled by the template engine (safety-critical —
    # must never be delegated to an external LLM).
    is_crisis = updated_state.get("crisis_flag") or prediction.emotion == "suicidal"

    if is_crisis:
        reply = _dm.get_next_reply(user_id, state=updated_state)
    else:
        # Try Groq LLM first; fall back to templates if unavailable
        try:
            reply = generate_reply(
                user_message=processing_text,
                emotion=prediction.emotion,
                confidence=prediction.confidence,
                history=_history[user_id],
            )
        except LLMUnavailable as e:
            # Print the FULL error so we can diagnose API key / network issues
            print("[pipeline] LLM failed - falling back to templates.")
            print(f"           Error: {e}")
            reply = _dm.get_next_reply(user_id, state=updated_state)

    # Track conversation history for LLM context (last 10 turns)
    _history[user_id].append({"role": "user",      "content": processing_text})
    _history[user_id].append({"role": "assistant", "content": reply})
    if len(_history[user_id]) > _MAX_HISTORY_TURNS:
        _history[user_id] = _history[user_id][-_MAX_HISTORY_TURNS:]

    translated_reply = translate_reply(reply, language_context, is_crisis=is_crisis)
    final_reply = translated_reply.reply

    # ── 8. Total latency ───────────────────────────────────────────────────
    total_latency_ms = (time.perf_counter() - t_pipeline_start) * 1000

    # ── 9. MongoDB logging (non-blocking) ──────────────────────────────────
    if mongo_db is not None:
        _log_to_mongo(
            db=mongo_db,
            user_id=user_id,
            raw_message=clean_text,      # sanitized, not raw
            processing_message=processing_text,
            emotion=prediction.emotion,
            confidence=prediction.confidence,
            reply=final_reply,
            latency_ms=total_latency_ms,
            language_context=language_context,
        )

    return {
        "reply": final_reply,
        "emotion": prediction.emotion,
        "confidence": round(prediction.confidence, 4),
        "latency_ms": round(total_latency_ms, 2),
        "cache_hit": prediction.cache_hit,
        "language": {
            "detected": language_context.detected_language,
            "reply": language_context.reply_language,
            "provider": language_context.provider,
            "input_translation_applied": language_context.input_translation_applied,
            "output_translation_applied": translated_reply.output_translation_applied,
            "input_translation_error": language_context.input_translation_error,
            "output_translation_error": translated_reply.output_translation_error,
            "detection_confidence": round(language_context.detection_confidence, 4),
        },
        "error": None,
        "status_code": 200,
    }
