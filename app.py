"""
app.py
------
Production Flask application for the Mental Health Chatbot NLP layer.

Key design decisions:
    - Model and connections are loaded ONCE at startup (not per-request)
    - Routes are thin — all logic lives in services/
    - /health endpoint for load balancer / monitoring checks
    - Session IDs are generated server-side (not trusted from client)
    - Gunicorn-compatible: no app.run() in production

Run in production:
    gunicorn -w 4 -b 0.0.0.0:5000 --timeout 60 app:app

Run in development:
    python app.py
"""

import uuid

# ── Load .env FIRST — before any other import reads os.getenv() ───────────────
try:
    from dotenv import load_dotenv
    # override=True  → .env values win over system environment variables
    # verbose=True   → prints which .env file was loaded (helps debug)
    loaded = load_dotenv(override=True, verbose=True)
    if not loaded:
        print("[app] ⚠️  .env file not found — falling back to system environment.")
except ImportError:
    pass

from flask import Flask, request, jsonify, render_template, session

from config.settings import settings
from services.nlp_pipeline import process_message
from utils.redis_pool import ping as redis_ping
from utils.logger import log_request, log_error


# ── MongoDB connection (optional — app works without it) ──────────────────────
try:
    from pymongo import MongoClient
    _mongo_client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
    _mongo_db = _mongo_client[settings.MONGO_DB_NAME]
    # Add indexes on first connect (idempotent)
    _mongo_db[settings.MONGO_COLLECTION].create_index("user_id")
    _mongo_db[settings.MONGO_COLLECTION].create_index("timestamp")
    _MONGO_AVAILABLE = True
except Exception as _e:
    _mongo_db = None
    _MONGO_AVAILABLE = False
    print(f"[app] MongoDB not available: {_e}. Continuing without persistence.")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = settings.SECRET_KEY

# ── CORS — allow the Next.js frontend to call this API ───────────────────────
# Development: http://localhost:3000
# Production:  https://compass-two-iota.vercel.app
try:
    from flask_cors import CORS
    CORS(app, origins=[
        "http://localhost:3000",
        "https://compass-two-iota.vercel.app",
        "https://compaass.vercel.app",
    ], supports_credentials=True)
    print("[app] CORS enabled for Next.js frontend.")
except ImportError:
    print("[app] flask-cors not installed — run: py -m pip install flask-cors")



# ── Helper: get or create session ID ─────────────────────────────────────────

def _get_user_id() -> str:
    """
    Return a stable session-scoped user ID.
    Generated server-side — never trusted from the client.
    """
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the chatbot UI."""
    return render_template("index.html")


@app.route("/send", methods=["POST"])
def send():
    """
    Main chatbot endpoint — called by the Next.js COMPASS frontend.

    Request body (JSON):
        {
            "message":    "I feel anxious",
            "session_id": "compass-conversation-id"   # optional — from Next.js
        }

    Response (JSON):
        {
            "reply":      "...",
            "emotion":    "anxiety",
            "confidence": 0.87
        }
    """
    import time
    t_start = time.perf_counter()

    # Parse request
    data = request.get_json(silent=True) or {}
    raw_text = data.get("message", "")

    # Prefer session_id from the request body (sent by Next.js per conversation).
    # This means each conversation thread gets its own isolated session context.
    # Fall back to the Flask cookie-based session for the legacy HTML UI.
    client_session_id = data.get("session_id", "").strip()
    user_id = client_session_id if client_session_id else _get_user_id()

    # Run full NLP pipeline
    result = process_message(
        raw_text=raw_text,
        user_id=user_id,
        mongo_db=_mongo_db,
    )

    latency_ms = (time.perf_counter() - t_start) * 1000
    log_request(
        user_id=user_id,
        route="/send",
        method="POST",
        status=result["status_code"],
        latency_ms=latency_ms,
    )

    response_body = {
        "reply":      result["reply"],
        "emotion":    result["emotion"],
        "confidence": result["confidence"],
    }
    if result["error"]:
        response_body["error"] = result["error"]

    return jsonify(response_body), result["status_code"]



@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Dialogflow fulfillment webhook.
    Receives Dialogflow POST, runs NLP pipeline, returns Dialogflow-format response.
    """
    user_id = _get_user_id()
    data = request.get_json(silent=True) or {}

    # Extract user message from Dialogflow request structure
    try:
        raw_text = data["queryResult"]["queryText"]
    except (KeyError, TypeError):
        return jsonify({"fulfillmentText": "I didn't catch that. Could you try again?"}), 200

    result = process_message(
        raw_text=raw_text,
        user_id=user_id,
        mongo_db=_mongo_db,
    )

    # Dialogflow expects this exact response structure
    return jsonify({"fulfillmentText": result["reply"]}), 200


@app.route("/health")
def health():
    """
    Health check endpoint for load balancers and monitoring tools.
    Returns 200 if the service is healthy, 503 if a critical dependency is down.
    """
    from models.emotion_classifier import classifier

    redis_ok = redis_ping()
    model_ok = classifier is not None

    status = {
        "status": "healthy" if (redis_ok and model_ok) else "degraded",
        "redis": "ok" if redis_ok else "unreachable",
        "model_loaded": model_ok,
        "mongo": "ok" if _MONGO_AVAILABLE else "unavailable",
    }

    http_status = 200 if (redis_ok and model_ok) else 503
    return jsonify(status), http_status


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def server_error(e):
    log_error(error=str(e), context="unhandled_exception")
    return jsonify({"error": "An internal error occurred. Please try again."}), 500


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Development only. In production, use:
    # gunicorn -w 4 -b 0.0.0.0:5000 --timeout 60 app:app
    app.run(
        host="0.0.0.0",
        port=settings.PORT,
        debug=(settings.FLASK_ENV == "development"),
    )
