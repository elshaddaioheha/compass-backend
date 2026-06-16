# COMPASS — Mental Health Chatbot NLP Backend

A production-style **Flask** backend for a mental health support chatbot. It detects the
**emotion** behind a user's message using a fine-tuned **DistilBERT** classifier, then
generates a warm, CBT-informed reply. A separate **crisis-detection layer** intercepts
high-risk messages (e.g. suicidal language) and responds with fixed, safe templates that
include Nigerian crisis hotlines.

The backend is multilingual: it accepts **English, Yoruba, and Nigerian Pidgin**, translating
non-English input to English for the classifier and translating the reply back to the user's
language.

> **For reviewers:** Section [How It Works](#how-it-works) explains the request flow end to
> end. Section [Quick Start](#quick-start) gets it running locally in a few minutes.

---

## How It Works

Every message flows through a single pipeline (`services/nlp_pipeline.py`). Each stage has one
job, which keeps the code easy to follow and test:

```
User message
    │
    ▼
1. InputValidator   reject empty / too-long input, strip HTML & control chars   (middleware/)
    │
    ▼
2. RateLimiter      block abuse via a Redis counter (fails open if Redis is down) (middleware/)
    │
    ▼
3. LanguageService  detect en / yo / pcm; translate non-English input to English  (services/)
    │
    ▼
4. Preprocessor     strip URLs, collapse "soooo" → "sooo", normalise whitespace   (services/)
    │
    ▼
5. EmotionClassifier DistilBERT (ONNX) → emotion label + confidence               (models/)
    │
    ▼
6. DialogueManager  update session state, detect crisis, choose reply strategy    (services/)
    │
    ├── crisis?  ──► fixed safe template + Nigerian hotlines  (never sent to the LLM)
    │
    └── otherwise ► Groq LLM empathetic reply, with a CBT template fallback        (services/)
    │
    ▼
7. LanguageService  translate the reply back to the user's language               (services/)
    │
    ▼
8. MongoDB (optional) persist the conversation record                             (app.py)
    │
    ▼
JSON response → { reply, emotion, confidence, language, session_id }
```

### Two design rules worth knowing

- **Crisis is never delegated to the LLM.** If the message trips the crisis layer (crisis
  keywords in English/Yoruba/Pidgin), the reply comes from a fixed, localized template so the
  emergency wording and hotline numbers stay exact. This is decided *before* the LLM is ever
  called (`services/nlp_pipeline.py`). A crisis turn also escalates to a human — see
  [Therapist Escalation](#therapist-escalation-emergency-alerts).
- **The app degrades gracefully.** Redis, MongoDB, and the Groq LLM are all optional. If Redis
  is down, the app uses an in-process cache and stops rate-limiting (fails open). If MongoDB is
  down, it simply skips persistence. If Groq is unavailable, it falls back to built-in CBT
  templates. The chatbot keeps answering in every case.

---

## Therapist Escalation (Emergency Alerts)

A chatbot should not be the only line of defense for someone in crisis. When the distress
layer flags a **hard crisis**, the backend escalates to a human: it emails an on-call therapist
so they can follow up, while the user simultaneously receives the crisis template with hotline
numbers (`services/alert_service.py`).

The escalation is built to be safe by construction:

- **Fail-safe.** The email is sent on a background thread and every error is swallowed and
  logged. Alerting can never delay, alter, or break the user's crisis reply.
- **De-duplicated.** At most one alert per user per `ALERT_COOLDOWN_SECONDS` (default 15 min),
  so an ongoing crisis conversation does not flood the therapist. Uses Redis when available,
  with an in-process fallback.
- **Opt-in.** Disabled unless `ENABLE_THERAPIST_ALERTS=true` and SMTP + a recipient are set.
  With it off, behavior is unchanged.

The alert email includes the session ID, timestamp, detected emotion + confidence, what
triggered the crisis, the user's message, and (for non-English) the English translation.

### Enabling alerts

Set these in `.env` (SMTP works with Gmail, SendGrid's SMTP relay, Mailgun, etc.):

```env
ENABLE_THERAPIST_ALERTS=true
THERAPIST_ALERT_EMAIL=oncall-therapist@example.com
ALERT_EMAIL_FROM=alerts@yourdomain.com        # defaults to SMTP_USERNAME
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=you@gmail.com
SMTP_PASSWORD=your-app-password               # Gmail app password, not your login password
SMTP_USE_TLS=true
ALERT_COOLDOWN_SECONDS=900
```

> **Privacy note:** an alert forwards the user's message to the configured therapist. That is
> appropriate and necessary for an emergency follow-up, but the recipient must be a real,
> authorized clinician and the channel should be one you trust.

---

## The Emotion Model

The classifier is a DistilBERT model fine-tuned on mental-health conversational data. It
predicts **five emotions**: `anger`, `anxiety`, `confusion`, `neutral`, `sadness`
(see `label_classes.json`).

> **Note:** "suicidal" and "depression" are *not* model classes. Suicidal-risk detection is
> handled separately by the keyword / language-marker crisis layer in
> `services/dialogue_manager.py` and `services/language_service.py`, which is more reliable for
> safety-critical wording than a learned label.

Predictions below `CONFIDENCE_THRESHOLD` (default `0.50`) are returned as `uncertain`, and the
bot asks a gentle clarifying question instead of guessing.

### Validation performance

Best epoch macro-F1 **0.9368** across **708** validation instances:

| Emotion     | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| `anger`     | 0.9655    | 0.9333 | 0.9492   | 150     |
| `anxiety`   | 0.8947    | 0.9067 | 0.9007   | 150     |
| `confusion` | 0.8992    | 0.9907 | 0.9427   | 108     |
| `neutral`   | 0.9603    | 0.9667 | 0.9635   | 150     |
| `sadness`   | 0.9574    | 0.9000 | 0.9278   | 150     |
| **accuracy**    |       |        | **0.9364** | 708   |
| **macro avg**   | 0.9354 | 0.9395 | 0.9368 | 708   |

For faster CPU inference the model is exported to **ONNX** and quantized (`convert_to_onnx.py`).
The full report is in `training_report.txt`.

---

## Project Structure

```
compass-backend/
├── app.py                     # Flask app: routes, CORS, Mongo wiring (Gunicorn-ready)
├── startup.py                 # Downloads the model from HuggingFace before boot
├── requirements.txt
├── .env.example               # Copy to .env and fill in values
├── render.yaml                # Render.com deployment blueprint (web + Redis)
├── label_classes.json         # The 5 emotion labels, in model output order
│
├── config/
│   └── settings.py            # All env-based config (no hardcoded secrets)
│
├── models/
│   └── emotion_classifier.py  # DistilBERT loader, ONNX inference, confidence gating, caching
│
├── services/
│   ├── nlp_pipeline.py        # Orchestrates the whole request flow (start here)
│   ├── language_service.py    # Language detection + translation + localized crisis replies
│   ├── preprocessor.py        # Text cleaning before the model
│   ├── dialogue_manager.py    # Redis-backed session state, crisis detection, CBT flows
│   ├── llm_service.py         # Groq LLM reply generation with safety prompt
│   └── alert_service.py       # Emergency therapist email alerts on crisis (fail-safe)
│
├── middleware/
│   ├── rate_limiter.py        # Redis sliding-window per-user rate limiting
│   └── input_validator.py     # Input sanitization + length checks
│
├── utils/
│   ├── logger.py              # Structured JSON logging
│   └── redis_pool.py          # Shared Redis connection pool (fast fail-open)
│
├── templates/
│   └── index.html             # Built-in chat UI (served at /)
│
├── tests/
│   └── test_nlp_pipeline.py   # 42 unit + integration tests (heavy deps mocked)
│
├── train.py                   # Fine-tune DistilBERT on the dataset
├── download_dataset.py        # Fetch / prepare the training dataset
├── convert_to_onnx.py         # Export + quantize the model to ONNX
└── upload_model.py            # Push the ONNX model to HuggingFace Hub
```

---

## Quick Start

**Requirements:** Python **3.12** (see `.python-version`). A Groq API key is recommended for
LLM replies. Redis and MongoDB are optional for local development.

### 1. Create the virtual environment and install dependencies

```bash
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (Git Bash):     source .venv/Scripts/activate
# macOS / Linux:          source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment variables

```bash
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
```

Then edit `.env`. The only values you usually need to provide are:

| Variable        | How to get it                                                                 |
|-----------------|-------------------------------------------------------------------------------|
| `GROQ_API_KEY`  | Create a key at <https://console.groq.com> → API Keys. Blank = template replies. |
| `HF_TOKEN`      | <https://huggingface.co> → Settings → Access Tokens. Only needed if the model repo is private. |
| `MONGO_URI`     | A MongoDB Atlas connection string, or leave the local default (persistence is optional). |
| `SECRET_KEY`    | Generate one: `python -c "import secrets; print(secrets.token_hex(32))"`.      |
| `REDIS_URL`     | `redis://localhost:6379/0` for a local Redis (optional).                       |

### 3. Download the emotion model

The model is hosted on HuggingFace (`HF_MODEL_REPO`, default `Oheha/compass-emotion-classifier`).
`startup.py` downloads it into `MODEL_DIR` (`./distilbert_finetuned`) if it isn't already there:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import startup; startup.download_model()"
```

### 4. Run locally

```bash
python app.py
# Serves on http://localhost:<PORT>  (PORT defaults to 5000; .env.example uses it as-is)
```

Open the URL in a browser for the built-in chat UI, or call the API directly (below).

### 5. Run in production

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload
```

Deployment to Render is described by `render.yaml` (a web service + a managed Redis instance);
`startup.py` runs during the build to fetch the model.

---

## Local Development Notes

- **Redis / MongoDB not running?** That's fine. `/health` will report `degraded` and the app
  uses in-process fallbacks. To silence Redis connection errors locally, point `REDIS_URL` at a
  local instance (`redis://localhost:6379/0`) or leave it unset.
- **`.env` ships production hosts.** If you copy a deployment `.env`, remember its `REDIS_URL` /
  `MONGO_URI` point at cloud services that may not be reachable from your machine. Use local
  values for local runs.
- **Port.** `config/settings.py` defaults `PORT` to `5000`; the Render blueprint and a deployed
  `.env` may set `10000`. Check your `.env` if the URL surprises you.

---

## Running Tests

All tests are self-contained — heavy dependencies (torch, redis, spacy, onnxruntime, pymongo)
are mocked, so no model or Redis instance is required.

```bash
python -m unittest tests/test_nlp_pipeline.py -v
```

`pytest` is not a project dependency, but the tests are pytest-compatible if you install it
(`pip install pytest && pytest tests/ -v`).

### Live language acceptance

This calls the configured translation provider, so run it only when `GROQ_API_KEY` is set:

```bash
python scripts/check_language_acceptance.py
```

See `docs/language_acceptance.md` for the manual Yoruba/Pidgin review checklist and
`docs/frontend_integration.md` for the external frontend request/response contract.

---

## API Endpoints

| Method | Route       | Description                                          |
|--------|-------------|------------------------------------------------------|
| GET    | `/`         | Built-in chat UI                                     |
| POST   | `/send`     | Send a message → `{ reply, emotion, confidence, ... }` |
| POST   | `/webhook`  | Dialogflow fulfillment webhook                       |
| GET    | `/health`   | Health check (Redis + model + Mongo status)          |

### Example `/send` request

```bash
curl -X POST http://localhost:5000/send \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel really anxious and cannot sleep", "language": "auto"}'
```

Optional request fields:

| Field              | Values                  | Purpose                                            |
|--------------------|-------------------------|----------------------------------------------------|
| `session_id`       | string                  | Stable conversation/session ID from the frontend   |
| `conversation_id`  | string                  | Alias for `session_id`                             |
| `language`         | `auto`, `en`, `yo`, `pcm` | User input language hint                          |
| `preferred_language` | `auto`, `en`, `yo`, `pcm` | Backward-compatible alias for `language`         |
| `reply_language`   | `en`, `yo`, `pcm`       | Force the bot reply language                        |

### Example response

```json
{
  "reply": "It sounds like you're feeling overwhelmed with anxiety right now. 💙 ...",
  "emotion": "anxiety",
  "confidence": 0.9888,
  "session_id": "compass-conversation-id",
  "language": {
    "detected": "en",
    "reply": "en",
    "provider": "groq",
    "input_translation_applied": false,
    "output_translation_applied": false
  }
}
```

---

## Configuration Reference

All configuration is read from environment variables (`config/settings.py`). The most relevant:

| Variable                          | Default                              | Purpose                                              |
|-----------------------------------|--------------------------------------|------------------------------------------------------|
| `FRONTEND_ORIGINS`                | localhost + Vercel URLs              | Comma-separated allowed CORS origins                 |
| `USE_ONNX`                        | `true`                               | Use the fast ONNX runtime (falls back to PyTorch)    |
| `MODEL_DIR` / `ONNX_MODEL_PATH`   | `./distilbert_finetuned` / ...       | Where the model + ONNX export live                   |
| `CONFIDENCE_THRESHOLD`            | `0.50`                               | Below this, the emotion is returned as `uncertain`   |
| `ENABLE_MULTILINGUAL`             | `true`                               | Enables language detection + translation             |
| `SUPPORTED_LANGUAGES`             | `en,yo,pcm`                          | Enabled language codes                               |
| `LANGUAGE_TRANSLATION_PROVIDER`   | `groq`                               | Translation provider (any other value disables it)   |
| `GROQ_MODEL`                      | `llama-3.1-8b-instant`               | Groq model for replies + translation                 |
| `LLM_HISTORY_TURNS`               | `6`                                  | Recent messages sent to the LLM for context          |
| `LLM_TIMEOUT_SECONDS`             | `8.0`                                | LLM timeout before template fallback                 |
| `RATE_LIMIT_REQUESTS` / `_WINDOW_SECONDS` | `30` / `60`                  | Per-user rate limit                                  |
| `ENABLE_THERAPIST_ALERTS`         | `false`                              | Master switch for emergency therapist email alerts   |
| `THERAPIST_ALERT_EMAIL`           | —                                    | On-call therapist recipient address                  |
| `SMTP_HOST` / `SMTP_PORT` / ...   | — / `587`                            | SMTP transport for alerts (Gmail, SendGrid, etc.)    |
| `ALERT_COOLDOWN_SECONDS`          | `900`                                | Max one therapist alert per user per window          |

### Infrastructure

| Service | Purpose                                              | Required? | Default            |
|---------|------------------------------------------------------|-----------|--------------------|
| Redis   | Session state, prediction cache, rate limiting       | Optional  | `localhost:6379`   |
| MongoDB | Conversation persistence                             | Optional  | `localhost:27017`  |
| Groq    | LLM replies + translation                            | Optional* | —                  |

\* Without Groq the bot still works using built-in CBT templates (English only).

---

## Status

- [x] Fine-tune DistilBERT on the mental-health dataset (`train.py`)
- [x] Evaluate model — per-class F1 reported above
- [x] ONNX export + quantization for fast CPU inference
- [x] Multilingual support (English / Yoruba / Nigerian Pidgin)
- [x] Crisis-detection safety layer with fixed localized templates
- [x] Emergency escalation: email an on-call therapist on crisis
- [ ] Add `Dockerfile` + `docker-compose.yml` for one-command local setup
- [ ] Deploy to cloud (Render blueprint in `render.yaml`)
