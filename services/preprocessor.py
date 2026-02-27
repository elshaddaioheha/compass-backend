"""
services/preprocessor.py
------------------------
Lightweight text preprocessing that runs BEFORE the DistilBERT tokenizer.

Why preprocess before tokenizing?
    - DistilBERT's tokenizer silently truncates at 512 tokens.
      If you feed 2000 characters, it silently drops the end.
      We hard-cap at MAX_INPUT_LENGTH characters first, explicitly.
    - Emoji and informal text are common in mental health conversations.
      We preserve them (DistilBERT handles unicode well) but normalize
      repeated characters ("I'm sooooo sad" → "I'm sooo sad").
    - URLs carry no emotional signal and waste token budget.

This module is intentionally stateless — no model, no Redis, no I/O.
It should be fast enough to be negligible in the latency budget.

Usage:
    from services.preprocessor import preprocess

    clean = preprocess("I'm feeling really really bad today 😢")
"""

import re
from config.settings import settings


# ── Compiled patterns (created once at module load) ───────────────────────────

# URLs: http/https/www links
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

# Repeated characters: "sooooo" → "sooo" (cap at 3 of the same char)
# This preserves "oo" in "good" but collapses "sooooooo"
_REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")

# Whitespace normalization: multiple spaces/tabs → single space
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")

# Multiple newlines → single newline
_NEWLINE_PATTERN = re.compile(r"\n+")


def preprocess(text: str) -> str:
    """
    Clean and normalize text for the DistilBERT tokenizer.

    Steps:
        1. Strip leading/trailing whitespace
        2. Remove URLs (no emotional signal, wastes tokens)
        3. Collapse repeated characters (e.g., "nooooo" → "nooo")
        4. Normalize whitespace
        5. Hard-cap at MAX_INPUT_LENGTH characters

    Args:
        text: A validated, sanitized string from input_validator.

    Returns:
        Preprocessed string ready for tokenization.
    """
    # 1. Strip
    text = text.strip()

    # 2. Remove URLs
    text = _URL_PATTERN.sub("", text)

    # 3. Collapse repeated characters (keeps up to 3 repetitions)
    text = _REPEATED_CHAR_PATTERN.sub(r"\1\1\1", text)

    # 4. Normalize whitespace
    text = _WHITESPACE_PATTERN.sub(" ", text)
    text = _NEWLINE_PATTERN.sub(" ", text)
    text = text.strip()

    # 5. Hard character cap — DistilBERT max is 512 tokens,
    #    roughly ~400 words / ~2000 chars, but we cap conservatively.
    #    We truncate at char level here; the tokenizer does its own token-level
    #    truncation, so this is a safety net, not the primary truncation.
    if len(text) > settings.MAX_INPUT_LENGTH:
        text = text[: settings.MAX_INPUT_LENGTH]

    return text
