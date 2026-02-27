"""
train.py
--------
Fine-tune DistilBERT for mental health emotion classification.

Expected dataset format (CSV):
    text,label
    "I can't stop worrying about everything",anxiety
    "I feel so empty and hopeless",depression
    ...

Supported emotion labels:
    anxiety | depression | anger | confusion | sadness | neutral | suicidal

Usage:
    # Basic (uses default paths)
    python train.py --data ./data/mental_health.csv

    # Full options
    python train.py \\
        --data      ./data/mental_health.csv \\
        --model-dir ./distilbert_finetuned \\
        --epochs    5 \\
        --batch     16 \\
        --lr        2e-5 \\
        --max-len   128 \\
        --val-split 0.15 \\
        --seed      42

Output:
    ./distilbert_finetuned/   — model + tokenizer files
    ./label_classes.json      — ordered label list (required by app)
    ./training_report.txt     — classification report (F1 per class)

After training, run:
    python convert_to_onnx.py --model-dir ./distilbert_finetuned --output-dir ./onnx_model
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    PyTorch Dataset for emotion classification.

    Args:
        texts:      List of raw text strings.
        labels:     List of integer label indices.
        tokenizer:  HuggingFace tokenizer.
        max_length: Max token length (default 128 — good balance for chat text).
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_validate(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset CSV and validate its structure.

    Expected columns: 'text' and 'label'.
    Rows with empty text or unknown labels are dropped with a warning.
    """
    print(f"\n[data] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Flexible column name handling
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"CSV must have 'text' and 'label' columns. Found: {list(df.columns)}"
        )

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"].str.len() > 0]

    after = len(df)
    if after < before:
        print(f"[data] Dropped {before - after} empty / null rows.")

    print(f"[data] Dataset size: {after} rows")
    print(f"[data] Label distribution:")
    for label, count in df["label"].value_counts().items():
        pct = count / after * 100
        print(f"         {label:<15} {count:>5}  ({pct:.1f}%)")

    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, class_weights):
    """Run one full training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    for batch in tqdm(loader, desc="  Train", leave=False):
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        # Gradient clipping — helps with DistilBERT stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, label_names):
    """
    Evaluate model on a DataLoader.
    Returns accuracy, macro-F1, and a full sklearn classification report.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    # Extract macro-F1 for early stopping
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]
    accuracy = report_dict["accuracy"]
    return accuracy, macro_f1, report


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Device: {device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    df = load_and_validate(args.data)

    # ── 2. Encode labels ──────────────────────────────────────────────────
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    label_names = le.classes_.tolist()
    num_labels  = len(label_names)

    print(f"\n[train] Labels ({num_labels}): {label_names}")

    # Save label classes — required by emotion_classifier.py
    labels_path = Path(args.model_dir).parent / "label_classes.json"
    with open(labels_path, "w") as f:
        json.dump(label_names, f, indent=2)
    print(f"[train] Saved label classes → {labels_path}")

    # ── 3. Train / validation split ───────────────────────────────────────
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_split,
        stratify=df["label_id"],
        random_state=args.seed,
    )
    print(f"\n[train] Train: {len(train_df)}, Val: {len(val_df)}")

    # ── 4. Tokenizer ──────────────────────────────────────────────────────
    print("\n[train] Loading tokenizer…")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # ── 5. Datasets & DataLoaders ─────────────────────────────────────────
    train_dataset = EmotionDataset(
        train_df["text"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        max_length=args.max_len,
    )
    val_dataset = EmotionDataset(
        val_df["text"].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
        max_length=args.max_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch, shuffle=False, num_workers=0)

    # ── 6. Class weights (handle imbalance) ───────────────────────────────
    class_weights = torch.tensor(
        compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_labels),
            y=train_df["label_id"].values,
        ),
        dtype=torch.float32,
    )
    print(f"[train] Class weights: { {label_names[i]: round(float(w), 3) for i, w in enumerate(class_weights)} }")

    # ── 7. Model ──────────────────────────────────────────────────────────
    print("\n[train] Loading DistilBERT…")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
    )
    model.to(device)

    # ── 8. Optimizer & scheduler ──────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = total_steps // 10   # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 9. Training loop ──────────────────────────────────────────────────
    print(f"\n[train] Starting training for {args.epochs} epoch(s)…")
    print(f"[train] Steps per epoch: {len(train_loader)}, Warmup: {warmup_steps}\n")

    best_f1      = 0.0
    best_epoch   = 0
    best_report  = ""
    patience     = args.patience
    no_improve   = 0

    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        print(f"── Epoch {epoch}/{args.epochs} ──")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, class_weights
        )
        acc, macro_f1, report = evaluate(model, val_loader, device, label_names)

        elapsed = time.time() - t_epoch
        print(f"  Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | {elapsed:.0f}s")

        # Save best model
        if macro_f1 > best_f1:
            best_f1     = macro_f1
            best_epoch  = epoch
            best_report = report
            no_improve  = 0
            model.save_pretrained(args.model_dir)
            tokenizer.save_pretrained(args.model_dir)
            print(f"  ✅ New best model saved (F1={best_f1:.4f}) → {args.model_dir}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\n[train] Early stopping triggered at epoch {epoch}.")
                break

        print()

    # ── 10. Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training complete. Best Macro-F1: {best_f1:.4f} (epoch {best_epoch})")
    print("=" * 60)
    print("\nClassification Report (best epoch):")
    print(best_report)

    # Save report to file
    report_path = "training_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best epoch: {best_epoch}  |  Best Macro-F1: {best_f1:.4f}\n\n")
        f.write(best_report)
    print(f"[train] Report saved → {report_path}")
    print(f"[train] Model saved  → {args.model_dir}/")
    print(f"\nNext step:")
    print(f"  python convert_to_onnx.py --model-dir {args.model_dir} --output-dir ./onnx_model")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for mental health emotion classification."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to training CSV (columns: text, label)",
    )
    parser.add_argument(
        "--model-dir", default="./distilbert_finetuned",
        help="Directory to save the fine-tuned model (default: ./distilbert_finetuned)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Max training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16; reduce to 8 if OOM)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max-len", type=int, default=128,
        help="Max token length (default: 128)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--patience", type=int, default=2,
        help="Early stopping patience in epochs (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    main(args)
