"""
startup.py
----------
Download the fine-tuned DistilBERT model from HuggingFace Hub if it doesn't
exist locally. Called automatically by Render's start command before gunicorn.

Set these environment variables on Render:
    HF_MODEL_REPO  = your-username/compass-emotion-classifier
    MODEL_DIR      = ./distilbert_finetuned
"""

import os
import sys

MODEL_DIR = os.getenv("MODEL_DIR", "./distilbert_finetuned")
HF_REPO   = os.getenv("HF_MODEL_REPO", "")
HF_TOKEN  = os.getenv("HF_TOKEN", "")   # Required when model repo is private

def download_model():
    """Download model from HuggingFace Hub if not already present locally."""
    # Check if model already exists (local dev or cached from previous deploy)
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        print(f"[startup] Model found at {MODEL_DIR} — skipping download.")
        return

    if not HF_REPO:
        print("[startup] ⚠️  HF_MODEL_REPO not set and no local model found.")
        print("           The classifier will fail to load.")
        print("           → Set HF_MODEL_REPO=your-hf-username/your-model-repo")
        return

    print(f"[startup] Downloading model from HuggingFace: {HF_REPO} ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN if HF_TOKEN else None,
        )
        print(f"[startup] ✅ Model downloaded to {MODEL_DIR}")
    except Exception as e:
        print(f"[startup] ❌ Model download failed: {e}")
        sys.exit(1)   # Fail loudly so Render shows the error



def download_spacy():
    """Download spaCy model if not already installed."""
    spacy_model = os.getenv("SPACY_MODEL", "en_core_web_sm")
    try:
        import spacy
        spacy.load(spacy_model)
        print(f"[startup] spaCy model '{spacy_model}' already installed.")
    except OSError:
        print(f"[startup] Downloading spaCy model: {spacy_model} ...")
        os.system(f"python -m spacy download {spacy_model}")
        print(f"[startup] ✅ spaCy model downloaded.")


if __name__ == "__main__":
    download_model()
    download_spacy()
    print("[startup] Ready. Starting gunicorn...")
