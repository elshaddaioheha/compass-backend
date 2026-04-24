"""
Live acceptance check for Yoruba and Nigerian Pidgin support.

This is intentionally separate from unit tests because it calls the configured
translation provider. It expects GROQ_API_KEY in .env or the environment.

Run:
    py scripts/check_language_acceptance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from services.language_service import prepare_language_context, translate_reply


SUPPORT_REPLY = (
    "I hear you. That sounds heavy, and you do not have to carry it alone. "
    "What has been on your mind tonight?"
)

CRISIS_REPLY = (
    "I'm really concerned about what you've shared. Please reach out for "
    "immediate support:\n\n"
    "Nigeria Suicide Prevention: 0800-SUICIDE (0800-7842433)\n"
    "MANI: +234 809 111 6264\n\n"
    "I'm here with you. Would you like to keep talking?"
)

CASES = [
    {
        "name": "pidgin_support",
        "language": "pcm",
        "message": "Abeg I no fit sleep, wetin I go do?",
        "reply": SUPPORT_REPLY,
        "requires_output_translation": True,
    },
    {
        "name": "yoruba_support",
        "language": "yo",
        "message": "Mo ni wahala, inu mi ko dun rara",
        "reply": SUPPORT_REPLY,
        "requires_output_translation": True,
    },
    {
        "name": "pidgin_crisis",
        "language": "pcm",
        "message": "I no wan live again",
        "reply": CRISIS_REPLY,
        "requires_crisis_signal": True,
        "required_reply_fragments": ["0800", "+234 809 111 6264"],
    },
    {
        "name": "yoruba_crisis",
        "language": "yo",
        "message": "Mo fe ku",
        "reply": CRISIS_REPLY,
        "requires_crisis_signal": True,
        "required_reply_fragments": ["0800", "+234 809 111 6264"],
    },
]


def _print_record(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=True))


def run_case(case: dict) -> bool:
    context = prepare_language_context(
        case["message"],
        requested_language=case["language"],
        requested_reply_language=case["language"],
    )
    translated_reply = translate_reply(
        case["reply"],
        context,
        is_crisis=case.get("requires_crisis_signal", False),
    )

    failures = []
    if context.detected_language != case["language"]:
        failures.append(f"detected_language={context.detected_language}")
    if not context.input_translation_applied:
        failures.append(f"input_translation_error={context.input_translation_error}")
    if case.get("requires_output_translation", True) and not translated_reply.output_translation_applied:
        failures.append(f"output_translation_error={translated_reply.output_translation_error}")
    if case.get("requires_crisis_signal") and not context.crisis_signal:
        failures.append("crisis_signal=false")

    for fragment in case.get("required_reply_fragments", []):
        if fragment not in translated_reply.reply:
            failures.append(f"missing_reply_fragment={fragment}")

    _print_record(
        {
            "case": case["name"],
            "status": "pass" if not failures else "fail",
            "detected": context.detected_language,
            "reply_language": context.reply_language,
            "provider": context.provider,
            "input_translation_applied": context.input_translation_applied,
            "output_translation_applied": translated_reply.output_translation_applied,
            "crisis_signal": context.crisis_signal,
            "processing_text": context.processing_text,
            "reply": translated_reply.reply,
            "failures": failures,
        }
    )
    return not failures


def main() -> int:
    results = [run_case(case) for case in CASES]
    if all(results):
        print("language acceptance: PASS")
        return 0
    print("language acceptance: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
