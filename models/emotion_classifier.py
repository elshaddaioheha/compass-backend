"""
models/emotion_classifier.py
-----------------------------
DistilBERT emotion classifier with:
    - ONNX Runtime inference (2–4x faster than PyTorch on CPU)
    - PyTorch fallback (if ONNX model not available)
    - Confidence thresholding (returns "uncertain" below threshold)
    - Redis prediction caching (avoids repeated model calls for same text)
    - Single model load at startup (not per-request)

The model is loaded ONCE when this module is first imported.
All subsequent calls share the same loaded model.

Emotion labels match your training dataset:
    anxiety, depression, anger, confusion, sadness, neutral, suicidal

Usage:
    from models.emotion_classifier import classifier

    result = classifier.predict("I feel completely hopeless")
    # → {"emotion": "depression", "confidence": 0.91, "cache_hit": False, "low_confidence": False}
"""

import hashlib
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from config.settings import settings
from utils.redis_pool import get_redis
from utils.logger import log_error

# ── Optional imports — graceful fallback ──────────────────────────────────────
try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    emotion: str
    confidence: float
    low_confidence: bool
    cache_hit: bool
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "confidence": round(self.confidence, 4),
            "low_confidence": self.low_confidence,
            "cache_hit": self.cache_hit,
            "latency_ms": round(self.latency_ms, 2),
        }


# ── Classifier ────────────────────────────────────────────────────────────────

class EmotionClassifier:
    """
    Wraps DistilBERT inference with caching and confidence gating.

    The constructor loads the model — call this once at app startup.
    """

    def __init__(self):
        self._labels: list[str] = self._load_labels()
        self._tokenizer = None
        self._ort_session = None
        self._torch_model = None
        self._use_onnx = settings.USE_ONNX and _ONNX_AVAILABLE
        self._load_model()
        self._warmup()   # pre-run one inference so the first real user request is fast

    # ── Label loading ─────────────────────────────────────────────────────

    def _load_labels(self) -> list[str]:
        """
        Load label classes from the JSON file saved during training.

        Expected format: ["anxiety", "depression", "anger", ...]

        If the file doesn't exist, falls back to a default set matching
        the emotion classes in your training dataset.
        """
        try:
            with open(settings.LABELS_PATH, "r") as f:
                labels = json.load(f)
            return labels
        except FileNotFoundError:
            # Fallback — matches the mental health dataset emotion classes
            return [
                "anxiety",
                "depression",
                "anger",
                "confusion",
                "sadness",
                "neutral",
                "suicidal",
            ]

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load tokenizer and model. Called once at startup."""
        if self._use_onnx:
            self._load_onnx()
        else:
            self._load_torch()

    def _load_onnx(self) -> None:
        """Load ONNX Runtime session for fast CPU inference."""
        try:
            # Use CPU provider; add "CUDAExecutionProvider" first if GPU is available
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 2  # tune for your server's CPU count

            self._ort_session = ort.InferenceSession(
                settings.ONNX_MODEL_PATH,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(settings.MODEL_DIR)
            print(f"[classifier] ONNX model loaded from {settings.ONNX_MODEL_PATH}")
        except Exception as e:
            log_error(error=str(e), context="onnx_load")
            print(f"[classifier] ONNX load failed ({e}), falling back to PyTorch.")
            self._use_onnx = False
            self._load_torch()

    def _load_torch(self) -> None:
        """Load PyTorch DistilBERT model as fallback, with INT8 quantization."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "Neither ONNX Runtime nor PyTorch/Transformers is installed. "
                "Cannot load the emotion classifier."
            )
        # Limit CPU threads — prevents PyTorch from spawning too many for short sequences
        torch.set_num_threads(2)

        self._tokenizer = DistilBertTokenizerFast.from_pretrained(settings.MODEL_DIR)
        self._torch_model = DistilBertForSequenceClassification.from_pretrained(
            settings.MODEL_DIR,
            num_labels=len(self._labels),
        )
        self._torch_model.eval()  # disable dropout for inference

        # INT8 dynamic quantization — compresses Linear weights to 8-bit
        # Gives ~2x speedup on CPU with minimal accuracy loss
        try:
            self._torch_model = torch.quantization.quantize_dynamic(
                self._torch_model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            print(f"[classifier] PyTorch model loaded + INT8 quantized from {settings.MODEL_DIR}")
        except Exception as e:
            # quantize_dynamic may fail on some platforms — fall back to fp32
            print(f"[classifier] INT8 quantization skipped ({e}), using fp32.")
            print(f"[classifier] PyTorch model loaded from {settings.MODEL_DIR}")

    # ── Inference ─────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> dict:
        """Tokenize text. Returns dict with input_ids and attention_mask."""
        return self._tokenizer(
            text,
            return_tensors="np" if self._use_onnx else "pt",
            truncation=True,
            # 128 tokens covers 99%+ of chat messages and is much faster than 512.
            # Transformer attention is O(n²) in sequence length, so 128 vs 512
            # is ~16x less computation in the attention layers.
            max_length=128,
            padding="max_length",
        )

    def _run_onnx(self, tokens: dict) -> np.ndarray:
        """Run ONNX Runtime inference. Returns raw logits array."""
        ort_inputs = {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
        }
        logits = self._ort_session.run(["logits"], ort_inputs)[0]
        return logits

    def _run_torch(self, tokens: dict) -> np.ndarray:
        """Run PyTorch inference. Returns raw logits as numpy array."""
        with torch.no_grad():
            outputs = self._torch_model(**tokens)
        return outputs.logits.numpy()

    def _logits_to_result(self, logits: np.ndarray) -> tuple[str, float]:
        """Convert raw logits → (emotion_label, confidence_probability)."""
        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        probs = probs.flatten()

        top_idx = int(probs.argmax())
        confidence = float(probs[top_idx])
        emotion = self._labels[top_idx] if top_idx < len(self._labels) else "neutral"
        return emotion, confidence

    # ── Cache ─────────────────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        """
        Deterministic cache key for a preprocessed text string.
        Using SHA256 to keep the key short and avoid Redis key-length issues.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"pred:{digest}"

    # ── In-process LRU cache (fallback when Redis is unavailable) ────────────
    # Keyed on text hash, capped at 256 entries (~few KB of memory).
    # This ensures repeated messages are instant even with no Redis.
    @lru_cache(maxsize=256)
    def _lru_predict(self, text: str) -> tuple:
        """LRU-cached raw inference (returns tuple for hashability)."""
        tokens = self._tokenize(text)
        if self._use_onnx:
            logits = self._run_onnx(tokens)
        else:
            logits = self._run_torch(tokens)
        emotion, confidence = self._logits_to_result(logits)
        return emotion, confidence

    def _get_cached(self, text: str) -> Optional[PredictionResult]:
        """Return Redis-cached prediction or None.
        
        Note: in-process LRU caching is handled automatically by
        @lru_cache on _lru_predict — no manual check needed here.
        """
        try:
            raw = get_redis().get(self._cache_key(text))
            if raw:
                data = json.loads(raw)
                return PredictionResult(
                    emotion=data["emotion"],
                    confidence=data["confidence"],
                    low_confidence=data["low_confidence"],
                    cache_hit=True,
                    latency_ms=0.0,
                )
        except Exception:
            pass  # Redis down — _lru_predict will serve from in-process cache
        return None

    def _set_cached(self, text: str, result: PredictionResult) -> None:
        """Store prediction in Redis with TTL."""
        try:
            key = self._cache_key(text)
            value = json.dumps({
                "emotion": result.emotion,
                "confidence": result.confidence,
                "low_confidence": result.low_confidence,
            })
            get_redis().setex(key, settings.PREDICTION_CACHE_TTL, value)
        except Exception as e:
            log_error(error=str(e), context="prediction_cache_write")

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, preprocessed_text: str) -> PredictionResult:
        """
        Predict emotion from preprocessed text.

        Args:
            preprocessed_text: Output of services/preprocessor.py preprocess().
                                Must already be cleaned and length-capped.

        Returns:
            PredictionResult with emotion, confidence, and metadata.
        """
        # 1. Check Redis cache first
        cached = self._get_cached(preprocessed_text)
        if cached:
            return cached

        # 2. Run model inference (with in-process LRU)
        t_start = time.perf_counter()
        try:
            emotion, confidence = self._lru_predict(preprocessed_text)
        except Exception as e:
            log_error(error=str(e), context="model_inference")
            emotion, confidence = "neutral", 0.0
        latency_ms = (time.perf_counter() - t_start) * 1000

        # 3. Confidence gating
        low_confidence = confidence < settings.CONFIDENCE_THRESHOLD
        if low_confidence:
            emotion = "uncertain"

        result = PredictionResult(
            emotion=emotion,
            confidence=confidence,
            low_confidence=low_confidence,
            cache_hit=False,
            latency_ms=latency_ms,
        )

        # 4. Write to Redis (non-blocking, optional)
        if not low_confidence:
            self._set_cached(preprocessed_text, result)

        return result

    def _warmup(self) -> None:
        """Run a dummy inference at startup so the first real user request is fast."""
        try:
            self._lru_predict("hello")
            print("[classifier] Warmup complete — model is ready.")
        except Exception as e:
            print(f"[classifier] Warmup failed (non-fatal): {e}")


# ── Singleton ─────────────────────────────────────────────────────────────────
# Loaded once when this module is first imported.
# All Flask routes and workers share this instance.
classifier = EmotionClassifier()
