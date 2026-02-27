"""
services/dialogue_manager.py
-----------------------------
Redis-backed Dialogue Manager for the mental health chatbot.

Responsibilities:
    1. Store and retrieve per-user session state from Redis (with TTL)
    2. Detect crisis signals (suicidal / high-risk language)
    3. Manage CBT exercise flows (step-by-step guidance)
    4. Generate contextually appropriate replies based on emotion + session state
    5. Log session lifecycle events

Session state schema (stored as JSON in Redis):
    {
        "last_emotion": "anxiety",
        "last_confidence": 0.87,
        "last_message": "I can't stop worrying",
        "cbt_active": false,
        "cbt_flow": null,          # e.g. "breathing" | "journaling" | "grounding"
        "cbt_step": 0,
        "cbt_offered": false,
        "crisis_flag": false,
        "turn_count": 4
    }

Usage:
    from services.dialogue_manager import DialogueManager
    dm = DialogueManager()

    dm.update_state(user_id, emotion="anxiety", confidence=0.87, message="...", entities={})
    reply = dm.get_next_reply(user_id)
"""

import json
import random
from typing import Optional

import spacy

from config.settings import settings
from utils.redis_pool import get_redis
from utils.logger import log_session, log_crisis, log_error

# ── spaCy — loaded once at module level ──────────────────────────────────────
try:
    _nlp = spacy.load(settings.SPACY_MODEL)
except OSError:
    _nlp = None
    print(f"[dialogue_manager] spaCy model '{settings.SPACY_MODEL}' not found. "
          "Run: python -m spacy download en_core_web_sm")


# ── Crisis keywords ───────────────────────────────────────────────────────────
# These are used as a secondary signal alongside the model's "suicidal" label.
# Kept minimal and focused on high-confidence crisis signals.
_CRISIS_KEYWORDS = frozenset([
    "kill myself", "end my life", "want to die", "suicide",
    "suicidal", "i can't go on", "no reason to live",
    "hurt myself", "self-harm", "cutting myself",
])

# ── Nigerian crisis resources (localized per your project scope) ──────────
_CRISIS_RESOURCES = (
    "Please reach out for help right now:\n"
    "🇳🇬 Nigeria Suicide Prevention: 0800-SUICIDE (0800-7842433)\n"
    "🇳🇬 MANI (Mental Awareness Nigeria Initiative): +234 809 111 6264\n"
    "💬 You can also text a trusted person or visit your nearest hospital."
)

# ── CBT flow templates ────────────────────────────────────────────────────────
_CBT_FLOWS = {
    "breathing": {
        "description": "a calming breathing exercise",
        "steps": [
            "Let's try a simple breathing exercise together. Find a comfortable position. 🧘",
            "Breathe in slowly through your nose for 4 counts. 1... 2... 3... 4...",
            "Hold your breath gently for 4 counts. 1... 2... 3... 4...",
            "Now breathe out slowly through your mouth for 6 counts. 1... 2... 3... 4... 5... 6...",
            "Great work. Repeat this 3 more times on your own. How do you feel now?",
        ],
    },
    "journaling": {
        "description": "a journaling prompt to help process your feelings",
        "steps": [
            "Writing can really help us process emotions. Let's do a quick journaling exercise. 📓",
            "Think about what's been weighing on you most today. What is the situation?",
            "Now ask yourself: what thoughts are you having about this situation?",
            "And how are those thoughts making you feel in your body right now?",
            "Is there another way you could view this situation? There's no right answer — just explore. 💭",
        ],
    },
    "grounding": {
        "description": "a grounding exercise to bring you back to the present",
        "steps": [
            "Let's try a grounding technique to help you feel more present. 🌿",
            "Look around you and name 5 things you can SEE right now.",
            "Now name 4 things you can physically TOUCH near you.",
            "Can you name 3 things you can HEAR around you right now?",
            "Name 2 things you can SMELL, and 1 thing you can TASTE. Well done — how do you feel?",
        ],
    },
}

# ── Response templates by emotion ────────────────────────────────────────────
_EMOTION_RESPONSES = {
    "anxiety": [
        "I can hear that you're feeling anxious. That takes a lot to share. 💙",
        "Anxiety can feel really overwhelming. I'm here with you.",
        "It sounds like you're carrying a lot of worry right now.",
    ],
    "depression": [
        "I'm really sorry you're feeling this way. You don't have to face it alone. 💙",
        "Depression can make everything feel heavy. I hear you.",
        "Thank you for trusting me with how you're feeling right now.",
    ],
    "anger": [
        "It sounds like something has really frustrated or upset you. That's valid.",
        "Anger often means something important to us feels threatened or unfair.",
        "I can hear how frustrated you are. Do you want to talk about what happened?",
    ],
    "confusion": [
        "Feeling confused and overwhelmed is completely understandable.",
        "Sometimes our minds just need a moment to untangle things. I'm here.",
        "Let's slow down together. What feels most unclear to you right now?",
    ],
    "sadness": [
        "I'm sorry you're feeling sad. Your feelings are valid and I'm listening. 🤍",
        "It's okay to feel sad. I'm here to sit with you through this.",
        "That sounds really painful. Thank you for opening up.",
    ],
    "neutral": [
        "I'm here and listening. How are you really doing today?",
        "It's good to check in. How have things been going for you?",
        "I'm here whenever you're ready to talk.",
    ],
    "uncertain": [
        "I want to make sure I understand you correctly. Could you tell me a bit more?",
        "I'm listening — can you share a little more about how you're feeling?",
        "I'm here for you. Can you tell me more about what's going on?",
    ],
}

_GREETINGS = frozenset([
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "hiya", "what's up", "sup",
])

_GREETING_RESPONSES = [
    "Hi there! I'm your Mental Health Companion. How are you feeling today? 😊",
    "Hello! I'm here to listen and support you. How are you doing right now?",
    "Hey! I'm glad you reached out. How are you feeling today?",
    "Hi! Welcome. This is a safe space. How are you feeling?",
]

_CBT_OFFER_TEMPLATES = {
    "anxiety": "Would you like to try {desc} that might help calm your nervous system?",
    "depression": "Would you like to try {desc} that could help you process what you're feeling?",
    "anger": "Would you like to try {desc} to help release some of that tension?",
    "sadness": "Would you like to try {desc} to gently explore your feelings?",
    "confusion": "Would you like to try {desc} to help bring some clarity?",
}

_EMOTION_TO_CBT = {
    "anxiety": "breathing",
    "depression": "journaling",
    "anger": "breathing",
    "sadness": "journaling",
    "confusion": "grounding",
}

_AFFIRMATIVE_WORDS = frozenset([
    "yes", "yeah", "yep", "sure", "ok", "okay", "alright",
    "let's go", "let's do it", "go ahead", "please", "why not",
])


# ── Default session state ─────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "last_emotion": "neutral",
        "last_confidence": 0.0,
        "last_message": "",
        "cbt_active": False,
        "cbt_flow": None,
        "cbt_step": 0,
        "cbt_offered": False,
        "crisis_flag": False,
        "turn_count": 0,
    }


# ── DialogueManager class ─────────────────────────────────────────────────────

class DialogueManager:
    """
    Stateless class — all state lives in Redis.
    Safe to instantiate once and share across workers.
    """

    def __init__(self):
        self._redis = get_redis()

    # ── Session helpers ───────────────────────────────────────────────────

    def _session_key(self, user_id: str) -> str:
        return f"session:{user_id}"

    def _load_state(self, user_id: str) -> dict:
        """Load session state from Redis. Returns default if not found."""
        try:
            raw = self._redis.get(self._session_key(user_id))
            if raw:
                return json.loads(raw)
            log_session(user_id=user_id, action="created")
            return _default_state()
        except Exception as e:
            log_error(error=str(e), context="session_read", user_id=user_id)
            return _default_state()

    def _save_state(self, user_id: str, state: dict) -> None:
        """Persist session state to Redis with TTL."""
        try:
            self._redis.setex(
                self._session_key(user_id),
                settings.SESSION_TTL_SECONDS,
                json.dumps(state),
            )
        except Exception as e:
            log_error(error=str(e), context="session_write", user_id=user_id)

    # ── Entity extraction ─────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> dict:
        """
        Use spaCy to extract duration, severity, and other relevant entities.
        Returns dict of extracted values (may be empty if spaCy not loaded).
        """
        if _nlp is None:
            return {}
        doc = _nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ in ("TIME", "DATE", "DURATION"):
                entities["duration"] = ent.text
            elif ent.label_ in ("CARDINAL", "ORDINAL"):
                entities["severity_hint"] = ent.text
        return entities

    # ── Crisis detection ──────────────────────────────────────────────────

    def _is_crisis(self, text: str, emotion: str) -> tuple[bool, str]:
        """
        Two-stage crisis check:
            1. Model detected "suicidal" emotion
            2. Keyword match in text

        Returns (is_crisis, triggered_by).
        """
        if emotion == "suicidal":
            return True, "model"
        text_lower = text.lower()
        for keyword in _CRISIS_KEYWORDS:
            if keyword in text_lower:
                return True, f"keyword:{keyword}"
        return False, ""

    # ── Greeting detection ────────────────────────────────────────────────

    def _is_greeting(self, text: str) -> bool:
        return text.strip().lower() in _GREETINGS

    # ── CBT flow helpers ──────────────────────────────────────────────────

    def _is_affirmative(self, text: str) -> bool:
        return any(word in text.lower() for word in _AFFIRMATIVE_WORDS)

    def _next_cbt_step(self, state: dict) -> tuple[str, dict]:
        """Advance the CBT flow by one step. Returns (reply, updated_state)."""
        flow_name = state["cbt_flow"]
        step = state["cbt_step"]
        flow = _CBT_FLOWS.get(flow_name)

        if not flow or step >= len(flow["steps"]):
            # Flow complete
            state["cbt_active"] = False
            state["cbt_flow"] = None
            state["cbt_step"] = 0
            reply = (
                "You've completed the exercise. 🌟 I'm proud of you for trying. "
                "How are you feeling now? I'm here if you want to talk more."
            )
        else:
            reply = flow["steps"][step]
            state["cbt_step"] = step + 1

        return reply, state

    # ── Public API ────────────────────────────────────────────────────────

    def update_state(
        self,
        user_id: str,
        emotion: str,
        confidence: float,
        message: str,
        entities: Optional[dict] = None,
    ) -> dict:
        """
        Update session state after a new user message.
        Returns the updated state dict.
        """
        state = self._load_state(user_id)
        state["last_emotion"] = emotion
        state["last_confidence"] = confidence
        state["last_message"] = message
        state["turn_count"] = state.get("turn_count", 0) + 1

        if entities:
            state["last_entities"] = entities

        # Check if user is agreeing to start a CBT exercise
        if state.get("cbt_offered") and self._is_affirmative(message):
            flow_name = _EMOTION_TO_CBT.get(emotion, "breathing")
            state["cbt_active"] = True
            state["cbt_flow"] = flow_name
            state["cbt_step"] = 0
        state["cbt_offered"] = False  # reset after each turn

        self._save_state(user_id, state)
        return state

    def get_next_reply(self, user_id: str, state: Optional[dict] = None) -> str:
        """
        Generate the next chatbot reply based on current session state.

        Args:
            user_id: The session user ID.
            state:   Optional — pass the dict returned by update_state() to
                     skip the Redis read entirely. This is critical when Redis
                     is unavailable, as re-reading would return default state
                     (neutral) and ignore the emotion we just detected.

        Priority order:
            1. Crisis → crisis resources (highest priority, overrides all else)
            2. Greeting → welcome message
            3. Active CBT flow → next step
            4. Uncertain emotion → clarifying question
            5. Emotional response + CBT offer
        """
        # Use the passed-in state (from update_state) when available,
        # otherwise fall back to loading from Redis.
        if state is None:
            state = self._load_state(user_id)
        message = state.get("last_message", "")
        emotion = state.get("last_emotion", "neutral")
        confidence = state.get("last_confidence", 0.0)

        # ── 1. Crisis check (highest priority) ────────────────────────────
        is_crisis, triggered_by = self._is_crisis(message, emotion)
        if is_crisis:
            state["crisis_flag"] = True
            state["cbt_active"] = False  # cancel any active CBT flow
            self._save_state(user_id, state)
            log_crisis(
                user_id=user_id,
                emotion=emotion,
                confidence=confidence,
                triggered_by=triggered_by,
            )
            return (
                "I'm really concerned about what you've shared and I want you to know "
                "you are not alone. 💙 Please reach out for immediate support:\n\n"
                + _CRISIS_RESOURCES
                + "\n\nI'm here with you. Would you like to keep talking?"
            )

        # ── 2. Greeting ────────────────────────────────────────────────────
        if self._is_greeting(message):
            log_session(user_id=user_id, action="resumed", cbt_active=state.get("cbt_active", False))
            return random.choice(_GREETING_RESPONSES)

        # ── 3. Active CBT flow ─────────────────────────────────────────────
        if state.get("cbt_active"):
            reply, state = self._next_cbt_step(state)
            self._save_state(user_id, state)
            return reply

        # ── 4. Low confidence / uncertain ─────────────────────────────────
        if emotion == "uncertain":
            return random.choice(_EMOTION_RESPONSES["uncertain"])

        # ── 5. Emotional response + CBT offer ─────────────────────────────
        empathy = random.choice(_EMOTION_RESPONSES.get(emotion, _EMOTION_RESPONSES["neutral"]))

        # Offer a CBT exercise if emotion warrants it and we haven't just offered one
        cbt_offer = ""
        if emotion in _EMOTION_TO_CBT and not state.get("cbt_offered"):
            flow_name = _EMOTION_TO_CBT[emotion]
            desc = _CBT_FLOWS[flow_name]["description"]
            template = _CBT_OFFER_TEMPLATES.get(emotion, "Would you like to try {desc}?")
            cbt_offer = "\n\n" + template.format(desc=desc)
            state["cbt_offered"] = True
            self._save_state(user_id, state)

        # Extract entities for richer context (e.g., "for two weeks")
        entities = self._extract_entities(message)
        duration_note = ""
        if "duration" in entities:
            duration_note = f"\n\nYou mentioned this has been going on for {entities['duration']}. " \
                            "That's a while to be carrying this."

        return empathy + duration_note + cbt_offer
