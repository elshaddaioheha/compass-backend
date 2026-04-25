# Project Context

This file is the lightweight working memory for this project. It lets us keep
chat context focused while still preserving the decisions and checks that matter.

## Current Shape

- Flask backend for a mental health chatbot NLP layer.
- Main API entrypoint is `app.py`.
- Core request path is:
  `InputValidator -> RateLimiter -> LanguageService -> Preprocessor -> EmotionClassifier -> DialogueManager`.
- The external frontend integration contract lives in `docs/frontend_integration.md`.
- Tests are concentrated in `tests/test_nlp_pipeline.py` and mock heavy dependencies.

## Active API Contract

- `POST /send` accepts `message`, optional `session_id`, optional
  `conversation_id`, optional `language`, and optional `reply_language`.
- `conversation_id` is treated as an alias for `session_id`.
- Response includes `reply`, `emotion`, `confidence`, and `session_id`.
- Response may include `language` metadata when multilingual handling is active.
- CORS origins are controlled by `FRONTEND_ORIGINS` without trailing slashes.

## Context Strategy

- Keep only the current task goal, touched files, and latest verification result
  in chat context.
- Reload exact files from disk when needed instead of relying on memory.
- Use this file for stable project facts and decisions.
- Use task-specific docs for detailed contracts, such as frontend integration or
  language acceptance.
- Avoid loading large artifacts such as model directories, `.env`, datasets, and
  generated model outputs unless the task explicitly requires them.

## Verification Checklist

Use the smallest check that proves the change:

- API contract or pipeline behavior: `python -m pytest tests/ -v`
- Language acceptance with live provider: `python scripts/check_language_acceptance.py`
- Local Flask smoke test: `python app.py`, then check `/health` and `/send`
- Documentation-only change: review the changed markdown and confirm links/commands

## Collaboration Notes

- Treat untracked or dirty files as user work unless we created them in the
  current task.
- Before code edits, explain the intended change and why it is the smallest safe
  move.
- Prefer existing project patterns over new abstractions.
- For risky changes, compare the main options and ask for approval before
  editing.
