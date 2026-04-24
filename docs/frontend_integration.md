# External Frontend Integration

This backend is designed for the COMPASS/Next.js frontend to call `/send`
directly from the browser.

## Environment

Set `FRONTEND_ORIGINS` on the backend to the exact browser origins that should
be allowed by CORS:

```text
FRONTEND_ORIGINS=http://localhost:3000,https://compass-two-iota.vercel.app,https://compaass.vercel.app
```

Do not include trailing slashes.

## Request

```ts
await fetch(`${BACKEND_URL}/send`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  credentials: "include",
  body: JSON.stringify({
    message,
    session_id: conversationId,
    language: "auto",          // or "en", "yo", "pcm"
    reply_language: "pcm",     // optional
  }),
});
```

`conversation_id` is also accepted as an alias for `session_id`.

## Response

```json
{
  "reply": "...",
  "emotion": "anxiety",
  "confidence": 0.88,
  "session_id": "conversation-123",
  "language": {
    "detected": "pcm",
    "reply": "pcm",
    "provider": "groq",
    "input_translation_applied": true,
    "output_translation_applied": true,
    "input_translation_error": null,
    "output_translation_error": null,
    "detection_confidence": 0.95
  }
}
```

The frontend can ignore `language` safely if it only needs the original
`reply`, `emotion`, and `confidence` fields.

## Compatibility Notes

- Existing payloads with only `{ "message": "..." }` still work.
- The backend returns `session_id` so a frontend can store it when it does not
  already own a conversation ID.
- Validation and rate-limit errors still return JSON with `error`.
- CORS preflight for `POST /send` is covered by tests for configured origins.
