# Mental Health Chatbot — Production NLP Layer

A Flask-based NLP backend for a mental health support chatbot. Uses a fine-tuned
**DistilBERT** model to detect emotions (anxiety, depression, anger, sadness, confusion,
suicidal ideation, neutral) from user messages and generates empathetic,
CBT-guided replies with crisis detection.

---

## Project Structure

```
final-year-project/
├── app.py                        # Flask app (Gunicorn-ready)
├── convert_to_onnx.py            # One-time ONNX export script
├── requirements.txt
├── .env.example                  # Copy to .env and fill in values
│
├── config/
│   ├── __init__.py
│   └── settings.py               # All env-based config (no hardcoded secrets)
│
├── models/
│   ├── __init__.py
│   └── emotion_classifier.py     # DistilBERT loader, ONNX inference, confidence gating
│
├── services/
│   ├── __init__.py
│   ├── language_service.py       # Language detection + translation wrapper
│   ├── preprocessor.py           # Text cleaning before model
│   ├── dialogue_manager.py       # Redis-backed session + CBT flow
│   └── nlp_pipeline.py           # Orchestrates preprocessor → model → DM
│
├── middleware/
│   ├── __init__.py
│   ├── rate_limiter.py           # Redis-based per-user rate limiting
│   └── input_validator.py        # Input sanitization + length checks
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Structured JSON logging with latency
│   └── redis_pool.py             # Shared connection pool
│
├── templates/
│   └── index.html                # Chat UI (served by Flask)
│
└── tests/
    ├── __init__.py
    └── test_nlp_pipeline.py      # Unit + integration tests (no heavy deps needed)
```

---

## Pipeline

```
User message
    │
    ▼
InputValidator      → rejects bad input (XSS, empty, too long)
    │
    ▼
RateLimiter         → blocks abuse (Redis sliding window)
    │
    ▼
LanguageService     → detects en/yo/pcm, translates to English when needed
    │
    ▼
Preprocessor        → strips URLs, normalises whitespace & repeated chars
    │
    ▼
EmotionClassifier   → DistilBERT / ONNX → emotion label + confidence score
    │
    ▼
DialogueManager     → CBT flow, crisis detection, empathetic reply (Redis-backed)
    │
    ▼
MongoDB logger      → persists conversation record (optional)
    │
    ▼
JSON response       → { reply, emotion, confidence }
```

Multilingual support is translation-first: Yoruba and Nigerian Pidgin input is
translated to English for the current DistilBERT classifier, then the bot reply
is translated back to the user's reply language when a translation provider is
available. If translation fails, the app falls back safely to the original text
or English reply instead of blocking the conversation.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in MONGO_URI, REDIS_URL, SECRET_KEY, MODEL_DIR, etc.
```

### 3. Provide the model

**Option A — Use a pre-trained fine-tuned model:**
Place your fine-tuned DistilBERT checkpoint in `./distilbert_finetuned/`
(must contain `config.json`, `pytorch_model.bin`, `tokenizer_config.json`, `vocab.txt`).

**Option B — Train from scratch:**
```bash
# (Once train.py is written) — see Next Steps
python train.py
```

### 4. Convert to ONNX (recommended for production)
```bash
python convert_to_onnx.py --model-dir ./distilbert_finetuned --output-dir ./onnx_model
# Update .env: ONNX_MODEL_PATH=./onnx_model/model_quantized.onnx
```

### 5. Run locally (development)
```bash
python app.py
# → http://localhost:5000
```

### 6. Run in production
```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 60 app:app
```

---

## Running Tests

All tests are self-contained — heavy dependencies (torch, redis, spacy) are mocked.
No model or Redis instance required.

```bash
# From the project root:
python -m pytest tests/ -v

# Or with unittest:
python -m unittest tests/test_nlp_pipeline.py -v
```

### Live language acceptance

This calls the configured translation provider, so run it only when `GROQ_API_KEY`
is available:

```bash
python scripts/check_language_acceptance.py
```

See `docs/language_acceptance.md` for the manual Yoruba/Pidgin review checklist.
See `docs/frontend_integration.md` for the external frontend request/response
contract.

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Chat UI |
| POST | `/send` | Send a message → `{ reply, emotion, confidence }` |
| POST | `/webhook` | Dialogflow fulfillment webhook |
| GET | `/health` | Health check (Redis + model status) |

### Example `/send` request
```bash
curl -X POST http://localhost:5000/send \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel really anxious and cannot sleep", "language": "auto"}'
```

Optional language fields:

| Field | Values | Purpose |
|-------|--------|---------|
| `session_id` | string | Stable conversation/session ID from the frontend |
| `conversation_id` | string | Alias for `session_id` if that fits the frontend naming better |
| `language` | `auto`, `en`, `yo`, `pcm` | User input language hint |
| `preferred_language` | `auto`, `en`, `yo`, `pcm` | Backward-compatible alias for `language` |
| `reply_language` | `en`, `yo`, `pcm` | Force bot reply language |

### Example response
```json
{
  "reply": "I can hear that you're feeling anxious. That takes a lot to share. 💙\n\nWould you like to try a calming breathing exercise that might help calm your nervous system?",
  "emotion": "anxiety",
  "confidence": 0.8912,
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

## Infrastructure Requirements

| Service | Purpose | Default |
|---------|---------|---------|
| Redis | Session state + prediction cache + rate limiting | `localhost:6379` |
| MongoDB | Conversation persistence (optional) | `localhost:27017` |

### Language configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `FRONTEND_ORIGINS` | `http://localhost:3000,https://compass-two-iota.vercel.app,https://compaass.vercel.app` | Comma-separated allowed CORS origins |
| `ENABLE_MULTILINGUAL` | `true` | Enables language detection and translation flow |
| `SUPPORTED_LANGUAGES` | `en,yo,pcm` | Enabled language codes |
| `LANGUAGE_TRANSLATION_PROVIDER` | `groq` | Translation provider, or any other value to disable provider translation |
| `LANGUAGE_TRANSLATION_TIMEOUT_SECONDS` | `8.0` | Translation request timeout |

---

## Next Steps

- [ ] Write `train.py` — fine-tune DistilBERT on mental health dataset
- [ ] Source / prepare dataset (e.g. mental health Reddit dataset from Kaggle)
- [ ] Add `Dockerfile` + `docker-compose.yml` for one-command local setup
- [ ] Deploy to cloud (Render / Railway / GCP)
- [ ] Evaluate model — report F1, accuracy per emotion class
