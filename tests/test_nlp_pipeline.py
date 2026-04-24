"""
tests/test_nlp_pipeline.py
--------------------------
Tests for each layer of the NLP pipeline.

All tests are self-contained using unittest.mock.
Heavy dependencies (spacy, redis, torch, onnxruntime, pymongo)
are mocked at the sys.modules level in setUpModule() so that
imports succeed without these packages being installed.

Run from the project root:
    python -m pytest tests/ -v
    # or
    python -m unittest tests/test_nlp_pipeline.py -v
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

# ── Add project root to path ──────────────────────────────────────────────────
# __file__ is <project_root>/tests/test_nlp_pipeline.py
# so the project root is one directory up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Mock all heavy/unavailable packages BEFORE any project import ─────────────
_MOCKED_MODULES = [
    "spacy", "redis", "redis.exceptions", "redis.connection",
    "pymongo", "pymongo.errors",
    "numpy",                                   # imported at top-level in emotion_classifier.py
    "torch", "transformers",
    "transformers.models", "transformers.models.distilbert",
    "onnxruntime", "onnxruntime.quantization",
]
for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ── 1. Input Validator Tests ───────────────────────────────────────────────────

class TestInputValidator(unittest.TestCase):

    def setUp(self):
        from middleware.input_validator import validate_input, InputValidationError
        self.validate = validate_input
        self.Error = InputValidationError

    def test_valid_message_passes(self):
        result = self.validate("I feel very anxious today")
        self.assertEqual(result, "I feel very anxious today")

    def test_strips_leading_trailing_whitespace(self):
        result = self.validate("  hello  ")
        self.assertEqual(result, "hello")

    def test_rejects_non_string(self):
        with self.assertRaises(self.Error):
            self.validate(12345)

    def test_rejects_empty_string(self):
        with self.assertRaises(self.Error):
            self.validate("")

    def test_rejects_whitespace_only(self):
        with self.assertRaises(self.Error):
            self.validate("   ")

    def test_strips_html_tags(self):
        result = self.validate("<script>alert(1)</script>Hello")
        self.assertNotIn("<script>", result)
        self.assertIn("Hello", result)

    def test_rejects_message_too_long(self):
        with self.assertRaises(self.Error):
            self.validate("x" * 1001)

    def test_normalizes_unicode(self):
        result = self.validate("Ａ")   # fullwidth A → ASCII A
        self.assertEqual(result, "A")

    def test_strips_control_characters(self):
        result = self.validate("hello\x00world")
        self.assertNotIn("\x00", result)


# ── 2. Preprocessor Tests ──────────────────────────────────────────────────────

class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        from services.preprocessor import preprocess
        self.preprocess = preprocess

    def test_removes_urls(self):
        result = self.preprocess("Check this out https://example.com please")
        self.assertNotIn("https://", result)
        self.assertIn("Check this out", result)

    def test_collapses_repeated_chars(self):
        result = self.preprocess("I'm sooooooo sad")
        self.assertNotIn("soooooo", result)
        self.assertIn("sooo", result)

    def test_normalizes_whitespace(self):
        result = self.preprocess("hello    world")
        self.assertEqual(result, "hello world")

    def test_truncates_long_text(self):
        long_text = "word " * 300
        result = self.preprocess(long_text)
        self.assertLessEqual(len(result), 512)

    def test_preserves_emoji(self):
        result = self.preprocess("I'm feeling sad 😢")
        self.assertIn("😢", result)

    def test_url_only_message_becomes_empty(self):
        result = self.preprocess("https://example.com")
        self.assertEqual(result.strip(), "")


# ── 3. Rate Limiter Tests ──────────────────────────────────────────────────────

class TestRateLimiter(unittest.TestCase):

    def _patch_redis(self, mock_redis):
        import middleware.rate_limiter as rl
        p = patch.object(rl, "get_redis", return_value=mock_redis)
        p.start()
        self.addCleanup(p.stop)

    def test_allows_requests_under_limit(self):
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [5, True]
        mock_redis.pipeline.return_value = mock_pipe
        self._patch_redis(mock_redis)
        from middleware.rate_limiter import check_rate_limit
        check_rate_limit("user_ok")   # should not raise

    def test_blocks_requests_over_limit(self):
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [31, True]  # over 30-request limit
        mock_redis.pipeline.return_value = mock_pipe
        self._patch_redis(mock_redis)
        from middleware.rate_limiter import check_rate_limit, RateLimitError
        with self.assertRaises(RateLimitError):
            check_rate_limit("user_over")

    def test_fails_open_on_redis_error(self):
        """If Redis is down, the user should NOT be blocked."""
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = Exception("Redis connection refused")
        self._patch_redis(mock_redis)
        from middleware.rate_limiter import check_rate_limit
        check_rate_limit("user_redis_down")   # should not raise


# ── 4. Dialogue Manager Tests ──────────────────────────────────────────────────

class TestDialogueManager(unittest.TestCase):

    def _make_dm(self, initial_state=None):
        """
        Create a DialogueManager backed by a stateful fake Redis.
        The fake Redis actually stores values so update_state → get_next_reply works.
        """
        _store = {}
        if initial_state:
            _store["session:pre"] = json.dumps(initial_state)

        def fake_get(key):
            return _store.get(key)

        def fake_setex(key, ttl, value):
            _store[key] = value

        mock_redis = MagicMock()
        mock_redis.get.side_effect = fake_get
        mock_redis.setex.side_effect = fake_setex

        import services.dialogue_manager as dm_mod
        p = patch.object(dm_mod, "get_redis", return_value=mock_redis)
        p.start()
        self.addCleanup(p.stop)
        from services.dialogue_manager import DialogueManager
        return DialogueManager()

    def test_greeting_returns_welcome_phrase(self):
        dm = self._make_dm()
        dm.update_state("u1", emotion="neutral", confidence=0.5, message="hello")
        reply = dm.get_next_reply("u1")
        # Any of the greeting responses contain one of these phrases
        self.assertTrue(
            any(phrase in reply.lower() for phrase in ["feeling", "listen", "glad", "welcome", "here"]),
            f"Unexpected greeting reply: {reply}"
        )

    def test_crisis_reply_contains_nigerian_hotline(self):
        dm = self._make_dm()
        dm.update_state("u2", emotion="suicidal", confidence=0.92, message="i want to die")
        reply = dm.get_next_reply("u2")
        self.assertIn("0800", reply)

    def test_crisis_interrupts_active_cbt_flow(self):
        dm = self._make_dm()
        dm.update_state("u3", emotion="suicidal", confidence=0.95, message="i want to die")
        reply = dm.get_next_reply("u3")
        self.assertIn("0800", reply)

    def test_uncertain_emotion_returns_clarification(self):
        dm = self._make_dm()
        dm.update_state("u4", emotion="uncertain", confidence=0.4, message="I don't know")
        reply = dm.get_next_reply("u4")
        self.assertGreater(len(reply), 0)

    def test_keyword_crisis_detection(self):
        """Keyword 'kill myself' triggers crisis even if model says 'neutral'."""
        dm = self._make_dm()
        dm.update_state("u5", emotion="neutral", confidence=0.6, message="I want to kill myself")
        reply = dm.get_next_reply("u5")
        self.assertIn("0800", reply)


# ── 5. Integration: process_message ───────────────────────────────────────────

class TestProcessMessage(unittest.TestCase):

    def setUp(self):
        from models.emotion_classifier import PredictionResult
        from services.language_service import LanguageContext, ReplyTranslation

        mock_prediction = PredictionResult(
            emotion="anxiety", confidence=0.88,
            low_confidence=False, cache_hit=False, latency_ms=42.0,
        )
        self._mock_classifier = MagicMock()
        self._mock_classifier.predict.return_value = mock_prediction

        self._mock_dm = MagicMock()
        self._mock_dm.update_state.return_value = {}
        self._mock_dm.get_next_reply.return_value = (
            "I hear you. Would you like to try a breathing exercise?"
        )

        import services.nlp_pipeline as pm
        self._LanguageContext = LanguageContext
        self._ReplyTranslation = ReplyTranslation
        self._p1 = patch.object(pm, "classifier", self._mock_classifier)
        self._p2 = patch.object(pm, "_dm", self._mock_dm)
        self._p3 = patch.object(
            pm,
            "prepare_language_context",
            side_effect=lambda text, requested_language=None, requested_reply_language=None: LanguageContext(
                original_text=text,
                processing_text=text,
                detected_language=requested_language or "en",
                reply_language=requested_reply_language or requested_language or "en",
                provider="test",
                input_translation_applied=False,
                input_translation_error=None,
                detection_confidence=1.0,
            ),
        )
        self._p4 = patch.object(
            pm,
            "translate_reply",
            side_effect=lambda reply, context, is_crisis=False: ReplyTranslation(
                reply=reply,
                output_translation_applied=False,
                output_translation_error=None,
            ),
        )
        self._p5 = patch.object(pm, "generate_reply", side_effect=pm.LLMUnavailable("LLM disabled in tests"))
        self._p1.start()
        self._p2.start()
        self._p3.start()
        self._p4.start()
        self._p5.start()

    def tearDown(self):
        self._p1.stop()
        self._p2.stop()
        self._p3.stop()
        self._p4.stop()
        self._p5.stop()

    def test_valid_message_returns_200_with_reply(self):
        from services.nlp_pipeline import process_message
        result = process_message("I feel really anxious right now", user_id="u_test")
        self.assertEqual(result["status_code"], 200)
        self.assertIsNone(result["error"])
        self.assertEqual(result["emotion"], "anxiety")
        self.assertIn("breathing", result["reply"])

    def test_empty_message_returns_400(self):
        from services.nlp_pipeline import process_message
        result = process_message("", user_id="u_test")
        self.assertEqual(result["status_code"], 400)
        self.assertIsNotNone(result["error"])

    def test_too_long_message_returns_400(self):
        from services.nlp_pipeline import process_message
        result = process_message("x" * 1001, user_id="u_test")
        self.assertEqual(result["status_code"], 400)

    def test_html_injection_is_stripped_not_errored(self):
        from services.nlp_pipeline import process_message
        result = process_message("<b>I feel anxious</b>", user_id="u_test")
        self.assertEqual(result["status_code"], 200)

    def test_rate_limited_user_returns_429(self):
        from middleware.rate_limiter import RateLimitError
        import services.nlp_pipeline as pm
        with patch.object(pm, "check_rate_limit",
                          side_effect=RateLimitError("Too many messages.")):
            from services.nlp_pipeline import process_message
            result = process_message("Hello", user_id="u_spammer")
        self.assertEqual(result["status_code"], 429)

    def test_result_contains_latency_field(self):
        from services.nlp_pipeline import process_message
        result = process_message("Feeling down", user_id="u_test")
        self.assertIn("latency_ms", result)
        self.assertIsInstance(result["latency_ms"], float)

    def test_multilingual_input_uses_translated_text_for_classifier(self):
        import services.nlp_pipeline as pm
        from services.nlp_pipeline import process_message

        pm.prepare_language_context.side_effect = lambda text, requested_language=None, requested_reply_language=None: self._LanguageContext(
            original_text=text,
            processing_text="I feel very anxious",
            detected_language="yo",
            reply_language="yo",
            provider="test",
            input_translation_applied=True,
            input_translation_error=None,
            detection_confidence=0.95,
        )
        pm.translate_reply.side_effect = lambda reply, context, is_crisis=False: self._ReplyTranslation(
            reply="Mo gbo pe ara re ko ya. Se o fe gbiyanju mimi die?",
            output_translation_applied=True,
            output_translation_error=None,
        )

        result = process_message("Ara mi n ya mi", user_id="u_yo", requested_language="yo")

        self.assertEqual(result["status_code"], 200)
        self._mock_classifier.predict.assert_called_with("I feel very anxious")
        self.assertEqual(result["language"]["detected"], "yo")
        self.assertTrue(result["language"]["input_translation_applied"])
        self.assertTrue(result["language"]["output_translation_applied"])
        self.assertIn("mimi", result["reply"])

    def test_language_crisis_signal_forces_template_reply(self):
        import services.nlp_pipeline as pm
        from services.nlp_pipeline import process_message

        pm.prepare_language_context.side_effect = lambda text, requested_language=None, requested_reply_language=None: self._LanguageContext(
            original_text=text,
            processing_text=text,
            detected_language="pcm",
            reply_language="pcm",
            provider="test",
            input_translation_applied=False,
            input_translation_error="provider unavailable",
            detection_confidence=0.95,
            crisis_signal=True,
            crisis_signal_source="language:pcm:i wan die",
        )

        result = process_message("I wan die", user_id="u_pcm", requested_language="pcm")

        self.assertEqual(result["status_code"], 200)
        self._mock_dm.get_next_reply.assert_called_once()
        _, kwargs = pm.translate_reply.call_args
        self.assertTrue(kwargs["is_crisis"])


class TestLanguageService(unittest.TestCase):

    def test_detects_pidgin_markers(self):
        from services.language_service import detect_language
        language, confidence = detect_language("Abeg I no fit sleep, wetin I go do?")
        self.assertEqual(language, "pcm")
        self.assertGreater(confidence, 0.5)

    def test_requested_language_overrides_heuristics(self):
        from services.language_service import detect_language
        language, confidence = detect_language("Hello there", requested_language="yoruba")
        self.assertEqual(language, "yo")
        self.assertEqual(confidence, 1.0)

    def test_detects_pidgin_crisis_signal(self):
        from services.language_service import detect_crisis_signal
        is_crisis, source = detect_crisis_signal("I wan die", "pcm")
        self.assertTrue(is_crisis)
        self.assertIn("pcm", source)

    def test_crisis_reply_uses_fixed_localized_template(self):
        from services.language_service import LanguageContext, translate_reply

        context = LanguageContext(
            original_text="Mo fe ku",
            processing_text="I want to die",
            detected_language="yo",
            reply_language="yo",
            provider="test",
            input_translation_applied=True,
            input_translation_error=None,
            detection_confidence=1.0,
            crisis_signal=True,
            crisis_signal_source="language:yo:mo fe ku",
        )

        result = translate_reply("English crisis reply", context, is_crisis=True)

        self.assertTrue(result.output_translation_applied)
        self.assertIn("0800-SUICIDE", result.reply)
        self.assertIn("+234 809 111 6264", result.reply)
        self.assertIn("Mo ni aniyan", result.reply)


class TestSendEndpoint(unittest.TestCase):

    def test_send_forwards_language_fields_and_returns_metadata(self):
        from app import app as flask_app
        import app as app_module

        expected_language = {
            "detected": "pcm",
            "reply": "pcm",
            "provider": "test",
            "input_translation_applied": True,
            "output_translation_applied": True,
            "input_translation_error": None,
            "output_translation_error": None,
            "detection_confidence": 0.95,
        }
        mock_result = {
            "reply": "I hear you",
            "emotion": "anxiety",
            "confidence": 0.88,
            "language": expected_language,
            "error": None,
            "status_code": 200,
        }

        with patch.object(app_module, "process_message", return_value=mock_result) as mock_process:
            client = flask_app.test_client()
            response = client.post(
                "/send",
                json={
                    "message": "I no fit sleep",
                    "session_id": "conversation-1",
                    "language": "pcm",
                    "reply_language": "pcm",
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["language"], expected_language)
        self.assertEqual(data["session_id"], "conversation-1")
        mock_process.assert_called_once()
        _, kwargs = mock_process.call_args
        self.assertEqual(kwargs["raw_text"], "I no fit sleep")
        self.assertEqual(kwargs["user_id"], "conversation-1")
        self.assertEqual(kwargs["requested_language"], "pcm")
        self.assertEqual(kwargs["requested_reply_language"], "pcm")

    def test_send_accepts_conversation_id_alias(self):
        from app import app as flask_app
        import app as app_module

        mock_result = {
            "reply": "I hear you",
            "emotion": "neutral",
            "confidence": 0.75,
            "error": None,
            "status_code": 200,
        }

        with patch.object(app_module, "process_message", return_value=mock_result) as mock_process:
            client = flask_app.test_client()
            response = client.post(
                "/send",
                json={
                    "message": "Hello",
                    "session_id": None,
                    "conversation_id": "thread-42",
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["session_id"], "thread-42")
        _, kwargs = mock_process.call_args
        self.assertEqual(kwargs["user_id"], "thread-42")

    def test_send_cors_preflight_allows_configured_frontend_origin(self):
        from app import app as flask_app

        client = flask_app.test_client()
        response = client.options(
            "/send",
            headers={
                "Origin": "https://compaass.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "https://compaass.vercel.app")
        self.assertIn("Content-Type", response.headers.get("Access-Control-Allow-Headers", ""))
        self.assertIn("POST", response.headers.get("Access-Control-Allow-Methods", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
