"""
convert_to_onnx.py
------------------
One-time script to export your fine-tuned DistilBERT model to ONNX format.

Run this ONCE after training is complete. The output ONNX model is then
used by the production app for 2–4x faster CPU inference.

What this does:
    1. Loads your fine-tuned DistilBERT (PyTorch)
    2. Traces it with a dummy input to produce a static computation graph
    3. Exports to ONNX format
    4. Applies INT8 quantization (cuts model size ~4x, speeds up ~2x further)
    5. Validates the output matches PyTorch output within tolerance
    6. Saves label classes to label_classes.json

Prerequisites:
    pip install optimum onnx onnxruntime

Usage:
    python convert_to_onnx.py --model-dir ./distilbert_finetuned --output-dir ./onnx_model

After running:
    Set USE_ONNX=true and ONNX_MODEL_PATH=./onnx_model/model_quantized.onnx in your .env
"""

import argparse
import json
import os
import sys

import numpy as np


def convert(model_dir: str, output_dir: str, quantize: bool = True) -> None:
    print(f"[convert] Loading PyTorch model from: {model_dir}")

    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    except ImportError:
        print("ERROR: torch and transformers must be installed.")
        sys.exit(1)

    # ── Load model and tokenizer ──────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    num_labels = model.config.num_labels
    print(f"[convert] Model loaded. Labels: {num_labels}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Save label classes ────────────────────────────────────────────────
    # If your training script used LabelEncoder, load it and save classes.
    # This is a placeholder — replace with your actual label list.
    labels_path = os.path.join(os.path.dirname(model_dir), "label_classes.json")
    if not os.path.exists(labels_path):
        print(f"[convert] WARNING: label_classes.json not found at {labels_path}.")
        print("          Creating placeholder. Edit it to match your training labels.")
        placeholder_labels = [f"class_{i}" for i in range(num_labels)]
        with open(labels_path, "w") as f:
            json.dump(placeholder_labels, f, indent=2)
        print(f"[convert] Saved placeholder labels to {labels_path}")
    else:
        print(f"[convert] Found existing label_classes.json at {labels_path}")

    # ── Create dummy input for tracing ────────────────────────────────────
    dummy_text = "I am feeling anxious and overwhelmed today"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,      # use shorter length for export; runtime uses 512
        padding="max_length",
    )

    # ── Export to ONNX ────────────────────────────────────────────────────
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"[convert] Exporting to ONNX: {onnx_path}")

    # PyTorch 2.4+ defaults to a new dynamo-based exporter that requires
    # the 'onnxscript' package (not yet available for Python 3.13).
    # Passing dynamo=False forces the legacy TorchScript exporter instead.
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    exported = False

    # Attempt 1: legacy TorchScript exporter (dynamo=False) — no onnxscript needed
    try:
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_path,
            dynamo=False,
            **export_kwargs,
        )
        exported = True
        print(f"[convert] ONNX export complete (legacy exporter): {onnx_path}")
    except TypeError:
        # Older PyTorch versions don't accept the dynamo kwarg — just omit it
        try:
            torch.onnx.export(
                model,
                (inputs["input_ids"], inputs["attention_mask"]),
                onnx_path,
                **export_kwargs,
            )
            exported = True
            print(f"[convert] ONNX export complete: {onnx_path}")
        except Exception as e2:
            print(f"[convert] torch.onnx.export failed: {e2}")
    except Exception as e:
        print(f"[convert] torch.onnx.export failed: {e}")

    # Attempt 2: HuggingFace optimum (most reliable for transformer models)
    if not exported:
        print("[convert] Falling back to HuggingFace optimum exporter…")
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                model_dir, export=True
            )
            ort_model.save_pretrained(output_dir)
            # optimum saves as model.onnx in the output dir
            if not os.path.exists(onnx_path):
                # some optimum versions use a subfolder name
                import glob
                found = glob.glob(os.path.join(output_dir, "**", "*.onnx"), recursive=True)
                if found:
                    onnx_path = found[0]
            exported = True
            print(f"[convert] optimum export complete: {onnx_path}")
        except ImportError:
            print("[convert] optimum not installed.")
            print("          Run: py -m pip install optimum[onnxruntime]")
        except Exception as e:
            print(f"[convert] optimum export failed: {e}")

    if not exported:
        print("\n[convert] ❌ ONNX export failed with all methods.")
        print("   Try: py -m pip install optimum[onnxruntime]")
        sys.exit(1)

    # ── Validate ONNX output ──────────────────────────────────────────────
    print("[convert] Validating ONNX output against PyTorch output...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy().astype(np.int64),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
        }
        ort_logits = session.run(["logits"], ort_inputs)[0]

        with torch.no_grad():
            pt_logits = model(**inputs).logits.numpy()

        max_diff = np.max(np.abs(pt_logits - ort_logits))
        print(f"[convert] Max logit difference (PyTorch vs ONNX): {max_diff:.6f}")
        if max_diff > 1e-3:
            print("[convert] WARNING: Difference is larger than expected. Check your model.")
        else:
            print("[convert] ✅ ONNX output matches PyTorch output.")
    except ImportError:
        print("[convert] onnxruntime not installed — skipping validation.")

    # ── INT8 Quantization ─────────────────────────────────────────────────
    if quantize:
        print("[convert] Applying INT8 dynamic quantization...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = os.path.join(output_dir, "model_quantized.onnx")
            quantize_dynamic(
                model_input=onnx_path,
                model_output=quantized_path,
                weight_type=QuantType.QInt8,
            )
            original_size = os.path.getsize(onnx_path) / 1024 / 1024
            quantized_size = os.path.getsize(quantized_path) / 1024 / 1024
            print(f"[convert] Original:   {original_size:.1f} MB")
            print(f"[convert] Quantized:  {quantized_size:.1f} MB")
            print(f"[convert] Size reduction: {(1 - quantized_size/original_size)*100:.0f}%")
            print(f"[convert] ✅ Quantized model saved: {quantized_path}")
            print(f"\n[convert] Set in your .env:")
            print(f"    ONNX_MODEL_PATH={quantized_path}")
        except ImportError:
            print("[convert] onnxruntime.quantization not available — skipping quantization.")
            print(f"\n[convert] Set in your .env:")
            print(f"    ONNX_MODEL_PATH={onnx_path}")
    else:
        print(f"\n[convert] Set in your .env:")
        print(f"    ONNX_MODEL_PATH={onnx_path}")

    print("\n[convert] Done. ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DistilBERT to ONNX")
    parser.add_argument(
        "--model-dir",
        default="./distilbert_finetuned",
        help="Path to your fine-tuned DistilBERT directory",
    )
    parser.add_argument(
        "--output-dir",
        default="./onnx_model",
        help="Directory to save ONNX model files",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization",
    )
    args = parser.parse_args()
    convert(args.model_dir, args.output_dir, quantize=not args.no_quantize)
