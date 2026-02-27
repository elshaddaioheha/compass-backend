"""
download_dataset.py
-------------------
Downloads the `dair-ai/emotion` dataset from HuggingFace and converts it
to the format expected by train.py (text, label columns).

Also attempts to merge with a local Kaggle mental health CSV if present.

Usage:
    py download_dataset.py

Output:
    ./data/mental_health.csv   — ready to pass to train.py

Prerequisites:
    py -m pip install datasets pandas scikit-learn
"""

import sys
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("./data")
OUT_PATH    = DATA_DIR / "mental_health.csv"
KAGGLE_PATH = DATA_DIR / "mental_health_corpus.csv"   # optional Kaggle file

# ── Label mappings ────────────────────────────────────────────────────────────
HF_LABEL_MAP = {
    "fear":     "anxiety",
    "sadness":  "sadness",
    "anger":    "anger",
    "joy":      "neutral",
    "love":     "neutral",
    "surprise": "confusion",
}

KAGGLE_LABEL_MAP = {
    "anxiety":              "anxiety",
    "depression":           "depression",
    "suicidal":             "suicidal",
    "stress":               "anxiety",
    "bipolar":              "depression",
    "personality disorder": "confusion",
    "normal":               "neutral",
}


# ── HuggingFace download ──────────────────────────────────────────────────────

def download_hf_dataset() -> pd.DataFrame:
    """Download dair-ai/emotion split-by-split (more reliable than combined)."""
    print("\n[download] Loading dair-ai/emotion from HuggingFace…")
    try:
        from datasets import load_dataset
    except ImportError:
        print("[download] ERROR: Run: py -m pip install datasets")
        return pd.DataFrame()

    frames = []
    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset("dair-ai/emotion", split=split, trust_remote_code=True)
            df = ds.to_pandas()

            # Robustly get label names — works with ClassLabel and plain int columns
            if "label" in ds.features and hasattr(ds.features["label"], "names"):
                label_names = ds.features["label"].names
                df["label_str"] = df["label"].apply(
                    lambda i: label_names[int(i)] if int(i) < len(label_names) else "unknown"
                )
            elif "label" in df.columns:
                df["label_str"] = df["label"].astype(str)
            else:
                print(f"[download] Split '{split}': no 'label' column. Columns: {list(df.columns)}")
                continue

            frames.append(df[["text", "label_str"]])
            print(f"[download]   {split}: {len(df)} rows")
        except Exception as e:
            print(f"[download] Split '{split}' failed: {e}")

    if not frames:
        print("[download] Could not load any HuggingFace splits.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = combined["label_str"].map(HF_LABEL_MAP)
    result = combined[["text", "label"]].dropna()
    result = result[result["text"].str.strip().str.len() > 3]

    print(f"\n[download] HuggingFace total: {len(result)} rows after mapping")
    _print_counts(result, "HuggingFace")
    return result


# ── Kaggle dataset (optional) ─────────────────────────────────────────────────

def load_kaggle_dataset() -> pd.DataFrame:
    """Load local Kaggle mental health CSV if present."""
    if not KAGGLE_PATH.exists():
        print(f"\n[download] Kaggle file not found at {KAGGLE_PATH} — skipping.")
        print(f"           Download from: https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus")
        return pd.DataFrame()

    print(f"\n[download] Loading Kaggle dataset from {KAGGLE_PATH}…")
    df = pd.read_csv(KAGGLE_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect text and label columns flexibly
    text_col  = next((c for c in df.columns if any(k in c for k in ["text", "statement", "post", "content"])), None)
    label_col = next((c for c in df.columns if any(k in c for k in ["label", "status", "tag", "class"])),    None)

    if not text_col or not label_col:
        print(f"[download] Cannot find text/label columns. Found: {list(df.columns)}")
        return pd.DataFrame()

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(KAGGLE_LABEL_MAP)
    df = df.dropna()
    df = df[df["text"].str.len() > 3]

    print(f"[download] Kaggle total: {len(df)} rows after mapping")
    _print_counts(df, "Kaggle")
    return df


# ── Balancing ─────────────────────────────────────────────────────────────────

def balance_dataset(df: pd.DataFrame, max_per_class: int = 1000) -> pd.DataFrame:
    """Cap each class at max_per_class rows. Works with all pandas versions."""
    parts = []
    for label in df["label"].unique():
        group = df[df["label"] == label]
        sample_size = min(len(group), max_per_class)
        parts.append(group.sample(sample_size, random_state=42))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_counts(df: pd.DataFrame, source: str) -> None:
    counts = df["label"].value_counts()
    print(f"  [{source} label breakdown]")
    for label, count in counts.items():
        bar = "█" * (count // 50)
        print(f"    {label:<15} {count:>5}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(exist_ok=True)

    hf_df     = download_hf_dataset()
    kaggle_df = load_kaggle_dataset()

    frames = [f for f in [hf_df, kaggle_df] if len(f) > 0]
    if not frames:
        print("\n[download] No data available. Exiting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["text", "label"])
    combined = combined[combined["text"].str.strip().str.len() > 5]

    combined = balance_dataset(combined, max_per_class=1000)

    print(f"\n[download] ── Final combined dataset ──────────────────")
    _print_counts(combined, "Final")
    print(f"  Total: {len(combined)} rows")

    combined.to_csv(OUT_PATH, index=False)
    print(f"\n[download] ✅ Saved → {OUT_PATH}")

    # Warn about missing labels
    all_labels   = {"anxiety", "depression", "anger", "confusion", "sadness", "neutral", "suicidal"}
    found_labels = set(combined["label"].unique())
    missing      = all_labels - found_labels
    if missing:
        print(f"\n⚠️  Missing labels (no training data): {missing}")
        print(f"   Consider downloading the Kaggle dataset for depression/suicidal coverage.")
        print(f"   Place it at: {KAGGLE_PATH}")

    print(f"\nNext step:")
    print(f"  py train.py --data {OUT_PATH} --epochs 5 --batch 16")


if __name__ == "__main__":
    main()
