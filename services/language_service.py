"""
services/language_service.py
----------------------------
Small language layer for multilingual chat support.

The classifier remains English-first. This module detects Yoruba / Nigerian
Pidgin, translates user input into English for the existing pipeline, and can
translate the final bot reply back to the user's language.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from config.settings import settings
from utils.logger import log_error


SUPPORTED_LANGUAGES = {
    "en": "English",
    "yo": "Yoruba",
    "pcm": "Nigerian Pidgin",
}

_LANGUAGE_ALIASES = {
    "english": "en",
    "en": "en",
    "yoruba": "yo",
    "yo": "yo",
    "pidgin": "pcm",
    "nigerian pidgin": "pcm",
    "naija": "pcm",
    "pcm": "pcm",
}

_YORUBA_MARKERS = (
    " emi ", " mo ", " mi ", " inu mi ", " okan mi ", " ibanuje ",
    " ebi ", " wahala mi ", " ko si ", " mo fe ", " mo n ", " se mi ",
    " ku ", " iku ", " ireti ", " orun ", " sugbon ",
)

_PIDGIN_MARKERS = (
    " dey ", " no fit ", " wetin ", " abeg ", " sabi ", " una ",
    " make i ", " make we ", " wahala ", " tire me ", " i tire ",
    " anyhow ", " sha ", " jare ", " e no ", " my body no ",
)

_YORUBA_CRISIS_MARKERS = (
    "mo fe ku", "mo fe pa ara mi",
    "iku dara", "ko si ireti", "emi ko le mo",
)

_PIDGIN_CRISIS_MARKERS = (
    "i wan die", "i want die", "i wan kill myself", "make i die",
    "i no wan live", "i no fit continue", "i no get reason to live",
)

_LOCALIZED_CRISIS_REPLIES = {
    "yo": (
        "Mo ni aniyan gan nipa ohun ti o so. Jowo wa iranlowo ni kiakia:\n\n"
        "Nigeria Suicide Prevention: 0800-SUICIDE (0800-7842433)\n"
        "MANI: +234 809 111 6264\n\n"
        "O ko nikan. Ba eniyan ti o gbekele soro tabi lo si ile iwosan to sun mo e. "
        "Mo wa nibi pelu re. Se o fe tesiwaju lati ba mi soro?"
    ),
    "pcm": (
        "I really dey concerned about wetin you share. Abeg reach out for help now:\n\n"
        "Nigeria Suicide Prevention: 0800-SUICIDE (0800-7842433)\n"
        "MANI: +234 809 111 6264\n\n"
        "You no dey alone. Call person wey you trust or go nearest hospital. "
        "I dey here with you. You wan continue to talk?"
    ),
}


class TranslationUnavailable(Exception):
    """Raised when the configured translation provider cannot translate."""


class TranslationProvider(Protocol):
    name: str

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        timeout: float,
    ) -> str:
        ...


@dataclass(frozen=True)
class LanguageContext:
    original_text: str
    processing_text: str
    detected_language: str
    reply_language: str
    provider: str
    input_translation_applied: bool
    input_translation_error: Optional[str]
    detection_confidence: float
    crisis_signal: bool = False
    crisis_signal_source: Optional[str] = None


@dataclass(frozen=True)
class ReplyTranslation:
    reply: str
    output_translation_applied: bool
    output_translation_error: Optional[str]


class NoopTranslationProvider:
    name = "none"

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        timeout: float,
    ) -> str:
        if source_language == target_language:
            return text
        raise TranslationUnavailable("No translation provider is configured.")


class GroqTranslationProvider:
    name = "groq"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise TranslationUnavailable("GROQ_API_KEY is not set.")
            try:
                from groq import Groq
            except ImportError as exc:
                raise TranslationUnavailable("groq package is not installed.") from exc
            self._client = Groq(api_key=api_key)
        return self._client

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        timeout: float,
    ) -> str:
        client = self._get_client()
        source_name = SUPPORTED_LANGUAGES.get(source_language, source_language)
        target_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
        model = os.getenv("LANGUAGE_TRANSLATION_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

        style_note = {
            "yo": (
                "Use natural, modern Yoruba that a Nigerian speaker can understand. "
                "Keep the tone warm and conversational. Avoid word-for-word literal output. "
                "For common support phrases, prefer: 'Mo gbo ohun ti o n so' for "
                "'I hear you', 'O dabi pe oro naa wuwo' for 'That sounds heavy', "
                "'Ko ye ki o koju re nikan' for 'you do not have to carry it alone', "
                "and 'Kini o wa lokan re lale yi?' for 'What has been on your mind tonight?'. "
                "Do not translate 'hear' as 'gbe'."
            ),
            "pcm": (
                "Use natural Nigerian Pidgin. Keep the tone warm and conversational. "
                "Avoid awkward or theatrical phrasing. Never write 'Meh hear you'; "
                "write 'I hear you' or 'I understand you'. Prefer 'Wetin dey your mind tonight?' "
                "for 'What has been on your mind tonight?'."
            ),
            "en": "Use natural, plain English.",
        }.get(target_language, "Use natural, plain language.")

        system_prompt = (
            "You are a careful translator for a mental health support app. "
            "Translate the user's text from the source language to the target language. "
            "Preserve hotline numbers, names, URLs, formatting, and line breaks. "
            f"{style_note} "
            "Return only the translation. Do not add advice or explanation."
        )
        user_prompt = (
            f"Source language: {source_name}\n"
            f"Target language: {target_name}\n\n"
            f"Text:\n{text}"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.1,
                timeout=timeout,
            )
            translated = response.choices[0].message.content.strip()
        except Exception as exc:
            raise TranslationUnavailable(str(exc)) from exc

        if not translated:
            raise TranslationUnavailable("Translation provider returned an empty response.")
        return translated


_provider_instance: Optional[TranslationProvider] = None


def _normalise_language_code(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    language = str(language).strip().lower()
    if language in ("", "auto"):
        return None
    return _LANGUAGE_ALIASES.get(language)


def _is_enabled_language(language: str) -> bool:
    return language in set(settings.SUPPORTED_LANGUAGES)


def _get_provider() -> TranslationProvider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider_name = settings.LANGUAGE_TRANSLATION_PROVIDER.strip().lower()
    if provider_name == "groq":
        _provider_instance = GroqTranslationProvider()
    else:
        _provider_instance = NoopTranslationProvider()
    return _provider_instance


def detect_language(text: str, requested_language: Optional[str] = None) -> tuple[str, float]:
    """
    Return a supported language code and rough confidence.

    This intentionally uses conservative heuristics. A real detector can replace
    this function later without touching the pipeline.
    """
    requested = _normalise_language_code(requested_language)
    if requested and _is_enabled_language(requested):
        return requested, 1.0

    normalized_text = re.sub(r"\s+", " ", text.lower()).strip()
    padded = f" {normalized_text} "
    yoruba_score = sum(1 for marker in _YORUBA_MARKERS if marker in padded)
    pidgin_score = sum(1 for marker in _PIDGIN_MARKERS if marker in padded)

    if yoruba_score > pidgin_score and _is_enabled_language("yo"):
        return "yo", min(0.95, 0.55 + yoruba_score * 0.15)
    if pidgin_score > 0 and _is_enabled_language("pcm"):
        return "pcm", min(0.95, 0.55 + pidgin_score * 0.15)
    return "en", 0.8


def detect_crisis_signal(text: str, language: str) -> tuple[bool, Optional[str]]:
    lowered = text.lower()
    markers = ()
    if language == "yo":
        markers = _YORUBA_CRISIS_MARKERS
    elif language == "pcm":
        markers = _PIDGIN_CRISIS_MARKERS

    for marker in markers:
        if marker in lowered:
            return True, f"language:{language}:{marker}"
    return False, None


def _translate(text: str, source_language: str, target_language: str) -> tuple[str, bool, Optional[str], str]:
    if source_language == target_language:
        provider = _get_provider()
        return text, False, None, provider.name

    provider = _get_provider()
    try:
        translated = provider.translate(
            text=text,
            source_language=source_language,
            target_language=target_language,
            timeout=settings.LANGUAGE_TRANSLATION_TIMEOUT_SECONDS,
        )
        return translated, True, None, provider.name
    except TranslationUnavailable as exc:
        error = str(exc)
        log_error(error=error, context="language_translation")
        return text, False, error, provider.name


def prepare_language_context(
    text: str,
    requested_language: Optional[str] = None,
    requested_reply_language: Optional[str] = None,
) -> LanguageContext:
    if not settings.ENABLE_MULTILINGUAL:
        return LanguageContext(
            original_text=text,
            processing_text=text,
            detected_language="en",
            reply_language="en",
            provider="disabled",
            input_translation_applied=False,
            input_translation_error=None,
            detection_confidence=1.0,
        )

    detected_language, confidence = detect_language(text, requested_language)
    reply_language = _normalise_language_code(requested_reply_language) or detected_language
    if not _is_enabled_language(reply_language):
        reply_language = detected_language

    crisis_signal, crisis_source = detect_crisis_signal(text, detected_language)
    processing_text, applied, error, provider = _translate(text, detected_language, "en")

    return LanguageContext(
        original_text=text,
        processing_text=processing_text,
        detected_language=detected_language,
        reply_language=reply_language,
        provider=provider,
        input_translation_applied=applied,
        input_translation_error=error,
        detection_confidence=confidence,
        crisis_signal=crisis_signal,
        crisis_signal_source=crisis_source,
    )


def translate_reply(reply: str, context: LanguageContext, is_crisis: bool = False) -> ReplyTranslation:
    if is_crisis and context.reply_language in _LOCALIZED_CRISIS_REPLIES:
        return ReplyTranslation(
            reply=_LOCALIZED_CRISIS_REPLIES[context.reply_language],
            output_translation_applied=True,
            output_translation_error=None,
        )

    if not settings.ENABLE_MULTILINGUAL or context.reply_language == "en":
        return ReplyTranslation(
            reply=reply,
            output_translation_applied=False,
            output_translation_error=None,
        )

    translated, applied, error, _provider = _translate(reply, "en", context.reply_language)
    return ReplyTranslation(
        reply=translated,
        output_translation_applied=applied,
        output_translation_error=error,
    )
