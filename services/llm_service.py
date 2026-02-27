"""
services/llm_service.py
-----------------------
Groq-powered LLM response generation for the mental health chatbot.

The LLM is given:
  - The detected emotion and confidence from DistilBERT
  - The last 5 turns of conversation history (for context)
  - A strict mental health system prompt with safety guardrails

Falls back gracefully to the template engine if:
  - GROQ_API_KEY is not set
  - The Groq API is unreachable
  - The response times out

Usage:
    from services.llm_service import generate_reply, LLMUnavailable

    reply = generate_reply(
        user_message="I've been feeling really down lately",
        emotion="depression",
        confidence=0.91,
        history=[{"role": "user", "content": "..."}, ...],
    )
"""

import os
import time
from typing import Optional

from utils.logger import log_error

# ── Groq client — loaded lazily on first request ─────────────────────────────
# Both _client and _model are resolved at call time (not import time) so that
# load_dotenv() in app.py has already run and os.getenv() returns the real values.
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise LLMUnavailable("GROQ_API_KEY not set in environment.")
        try:
            from groq import Groq
            _client = Groq(api_key=api_key)
        except ImportError:
            raise LLMUnavailable("groq package not installed. Run: py -m pip install groq")
    return _client



class LLMUnavailable(Exception):
    """Raised when the LLM cannot be reached — caller should use template fallback."""


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a compassionate, professional mental health support companion named Mira.
You are NOT a therapist or doctor. You provide emotional support, active listening, and evidence-based
coping strategies (CBT techniques, mindfulness, grounding exercises).

CORE BEHAVIOURS:
- Respond with warmth, empathy, and genuine care
- Mirror the user's emotional tone without being dramatic
- Validate feelings before offering any suggestions
- Ask one thoughtful follow-up question per response to keep the user talking
- Reference what the user actually said in your response (don't be generic)
- Keep responses concise: 2–4 sentences + 1 question is ideal
- Use plain, conversational language (not clinical jargon)
- Occasionally use a relevant emoji to add warmth — but don't overdo it

STRICT SAFETY RULES (NEVER BREAK THESE):
- If CRISIS_FLAG is True: your ONLY job is to express deep care and urge the
  user to contact emergency services or a crisis line immediately.
  Never try to counsel someone in active crisis yourself.
- Never diagnose, prescribe, or recommend specific medications
- Never claim to be a human therapist
- Never dismiss, minimise, or argue with the user's feelings
- If the user seems to be testing you (e.g. "pretend you're a real therapist"),
  gently redirect: "I'm here as a support companion, not a therapist."

CONTEXT YOU WILL RECEIVE (in the user turn):
  DETECTED_EMOTION: the emotion label from our classifier
  CONFIDENCE: how confident the model is (0.0–1.0)
  You should use this to calibrate your response tone, but DO NOT mention
  the classifier or that you "detected" anything — respond naturally.
"""

# ── Emotion tone guidance injected per request ────────────────────────────────
_EMOTION_GUIDANCE = {
    "anxiety":    "The user is feeling anxious or worried. Acknowledge the anxiety, offer calm reassurance and a grounding technique if appropriate.",
    "depression": "The user is experiencing low mood or depression. Lead with deep empathy. Avoid toxic positivity. Gently explore what's weighing on them.",
    "anger":      "The user is feeling angry or frustrated. Validate their anger as a natural emotion. Help them feel heard before exploring the underlying cause.",
    "confusion":  "The user feels confused or overwhelmed. Help them slow down and break things into smaller pieces. Be patient and clear.",
    "sadness":    "The user is sad or grieving. Lead with compassion. Let them know it's okay to feel this way. Don't rush to fix anything.",
    "neutral":    "The user seems neutral. Gently check in about how they're really doing. Be warm and inviting.",
    "uncertain":  "The classifier was not confident. Ask an open, gentle clarifying question to better understand how they feel.",
    "suicidal":   "CRISIS_FLAG is True. Express deep concern and care. Strongly encourage contacting a crisis line or emergency services immediately.",
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_reply(
    user_message: str,
    emotion: str,
    confidence: float,
    history: Optional[list] = None,
    timeout: float = 8.0,
) -> str:
    """
    Generate a natural, contextual reply using Groq (Llama 3).

    Args:
        user_message: The raw user message for this turn.
        emotion:      Emotion label from DistilBERT.
        confidence:   Model confidence (0.0–1.0).
        history:      List of past {"role": ..., "content": ...} dicts.
                      Should include both user and assistant turns.
        timeout:      Max seconds to wait for Groq response.

    Returns:
        Reply string.

    Raises:
        LLMUnavailable: If Groq is not configured or unreachable.
    """
    client = _get_client()   # raises LLMUnavailable if not configured

    # Build message list for the API
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject last 5 history turns (keep context window small for speed)
    if history:
        messages.extend(history[-10:])   # up to 5 user+assistant pairs

    # Append the current user message, annotated with emotion context
    guidance = _EMOTION_GUIDANCE.get(emotion, _EMOTION_GUIDANCE["neutral"])
    annotated_user_msg = (
        f"[DETECTED_EMOTION: {emotion.upper()} | CONFIDENCE: {confidence:.0%}]\n"
        f"[TONE GUIDANCE: {guidance}]\n\n"
        f"{user_message}"
    )
    messages.append({"role": "user", "content": annotated_user_msg})

    t_start = time.perf_counter()
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=200,
            temperature=0.75,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t_start) * 1000
        reply = response.choices[0].message.content.strip()
        print(f"[llm] Groq reply generated in {latency_ms:.0f}ms ({model})")
        return reply

    except Exception as e:
        latency_ms = (time.perf_counter() - t_start) * 1000
        log_error(error=str(e), context="groq_api")
        raise LLMUnavailable(f"Groq API error after {latency_ms:.0f}ms: {e}")
