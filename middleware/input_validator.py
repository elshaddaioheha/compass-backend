"""
middleware/input_validator.py
------------------------------
Validates and sanitizes every user message before it touches the NLP model.

Checks performed (in order):
    1. Type check — must be a string
    2. Empty check — must not be blank
    3. Length check — must not exceed MAX_RAW_CHARS
    4. Sanitization — strips HTML tags, control characters, null bytes
    5. Unicode normalization — NFKC (e.g. fullwidth chars → ASCII)

Returns a clean string on success, or raises InputValidationError with
a user-friendly message that the Flask route can return directly.

Usage:
    from middleware.input_validator import validate_input, InputValidationError

    try:
        clean_text = validate_input(raw_text)
    except InputValidationError as e:
        return jsonify({"error": str(e)}), 400
"""

import re
import unicodedata
from config.settings import settings


class InputValidationError(ValueError):
    """Raised when user input fails validation. Message is safe to expose."""
    pass


# Pre-compiled patterns for performance (compiled once at import time)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # keeps \t \n \r
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


def _strip_html(text: str) -> str:
    return _HTML_TAG_PATTERN.sub("", text)


def _strip_control_chars(text: str) -> str:
    return _CONTROL_CHAR_PATTERN.sub("", text)


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE_PATTERN.sub(" ", text)
    text = _MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def _normalize_unicode(text: str) -> str:
    # NFKC: converts fullwidth letters, ligatures, etc. to standard forms
    return unicodedata.normalize("NFKC", text)


def validate_input(raw_text) -> str:
    """
    Validate and sanitize raw user input.

    Args:
        raw_text: The value received from the request (any type).

    Returns:
        Clean, safe string ready for the NLP pipeline.

    Raises:
        InputValidationError: If validation fails (message is user-safe).
    """
    # 1. Type check
    if not isinstance(raw_text, str):
        raise InputValidationError("Message must be a text string.")

    # 2. Length check BEFORE sanitization (saves processing malicious payloads)
    if len(raw_text) > settings.MAX_RAW_CHARS:
        raise InputValidationError(
            f"Message is too long. Please keep it under {settings.MAX_RAW_CHARS} characters."
        )

    # 3. Sanitize
    text = _normalize_unicode(raw_text)
    text = _strip_html(text)
    text = _strip_control_chars(text)
    text = _normalize_whitespace(text)

    # 4. Empty check AFTER sanitization (e.g. input was just "<script></script>")
    if not text:
        raise InputValidationError("Message cannot be empty.")

    return text
