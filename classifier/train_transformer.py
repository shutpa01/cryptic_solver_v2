"""
Fine-tune DistilBERT for cryptic crossword clue type classification.

Usage:
    python classifier/train_transformer.py                              # Train + evaluate + save
    python classifier/train_transformer.py --epochs 5 --batch-size 16   # Custom params
    python classifier/train_transformer.py --evaluate                   # Evaluate saved model
    python classifier/train_transformer.py --predict "clue" ANSWER 7    # Predict single clue
    python classifier/train_transformer.py --quick                      # Train on 10% for testing
"""

import argparse
import functools
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Force unbuffered output so background runs show progress
print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "clues_master.db"
MODEL_DIR = Path(__file__).resolve().parent / "transformer_model"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClueDataset(Dataset):
    """Tokenised clue dataset for PyTorch DataLoader."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(quick: bool = False):
    """Load labelled clues from clues_master.db.

    Returns (texts_a, texts_b, labels_str) where:
        texts_a = list of clue texts
        texts_b = list of "ANSWER (enum)" strings
        labels_str = list of wordplay_type strings
    """
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute(
        "SELECT clue_text, answer, enumeration, wordplay_type "
        "FROM clues "
        "WHERE wordplay_type IS NOT NULL AND wordplay_type != ''"
    )
    rows = cur.fetchall()
    conn.close()

    texts_a, texts_b, labels = [], [], []
    for clue_text, answer, enum, wtype in rows:
        if not clue_text or not wtype:
            continue
        texts_a.append(clue_text.strip())
        ans_part = (answer or "").strip().upper()
        enum_part = f"({enum})" if enum else ""
        texts_b.append(f"{ans_part} {enum_part}".strip())
        labels.append(wtype.strip().lower())

    # Group rare classes (<500 samples) into 'other'
    from collections import Counter
    counts = Counter(labels)
    labels = [l if counts[l] >= 500 else "other" for l in labels]

    if quick:
        # Stratified 10% sample
        _, texts_a, _, texts_b, _, labels = train_test_split(
            texts_a, texts_b, labels,
            test_size=0.10, random_state=42, stratify=labels,
        )

    print(f"Loaded {len(labels):,} labelled clues")
    counts = Counter(labels)
    for cls in sorted(counts, key=counts.get, reverse=True):
        print(f"  {cls:20s} {counts[cls]:>6,}")

    return texts_a, texts_b, labels


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenize(tokenizer, texts_a, texts_b, max_length=64):
    """Tokenise with [CLS] clue [SEP] ANSWER (enum) [SEP]."""
    return tokenizer(
        texts_a,
        texts_b,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Data ---
    texts_a, texts_b, labels_str = load_data(quick=args.quick)

    # Build label map
    classes = sorted(set(labels_str))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    labels_int = [label2id[l] for l in labels_str]

    # Stratified split
    (train_a, test_a, train_b, test_b, train_labels, test_labels) = train_test_split(
        texts_a, texts_b, labels_int,
        test_size=0.20, random_state=42, stratify=labels_int,
    )
    print(f"\nTrain: {len(train_labels):,}  Test: {len(test_labels):,}")

    # --- Tokenise ---
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_enc = tokenize(tokenizer, train_a, train_b, max_length=args.max_length)
    test_enc = tokenize(tokenizer, test_a, test_b, max_length=args.max_length)

    train_dataset = ClueDataset(train_enc, torch.tensor(train_labels, dtype=torch.long))
    test_dataset = ClueDataset(test_enc, torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # --- Class weights ---
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

    # --- Model ---
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # --- Optimiser + scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # --- Training loop ---
    print(f"\nTraining for {args.epochs} epochs ({total_steps:,} steps, "
          f"warmup {warmup_steps:,})...\n")
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if step % 25 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} step {step:>4}/{len(train_loader)} "
                      f"loss={loss.item():.4f}  lr={lr:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # --- Evaluate after each epoch ---
        acc, report_str, _ = evaluate_model(model, test_loader, id2label, device, loss_fn)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.4f}  "
              f"test_acc={acc:.4f}  elapsed={elapsed:.0f}s")
        print(f"{'='*60}\n")

        if acc > best_acc:
            best_acc = acc
            # Save best model
            save_model(model, tokenizer, id2label, label2id, best_acc, args, MODEL_DIR)
            print(f"  >> Saved best model (acc={best_acc:.4f})\n")

    # --- Final evaluation on best model ---
    print(f"\n{'='*60}")
    print(f"FINAL — Best test accuracy: {best_acc:.4f}")
    print(f"{'='*60}")

    # Reload best and show full report
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    acc, report_str, cm = evaluate_model(model, test_loader, id2label, device, loss_fn)
    print(f"\n{report_str}")
    print(f"\nConfusion matrix:\n{cm}")


def evaluate_model(model, dataloader, id2label, device, loss_fn=None):
    """Run evaluation, return (accuracy, classification_report_str, confusion_matrix)."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    target_names = [id2label[i] for i in range(len(id2label))]
    report_str = classification_report(
        all_labels, all_preds, target_names=target_names, digits=3, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report_str, cm


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(model, tokenizer, id2label, label2id, accuracy, args, model_dir):
    """Save model, tokenizer, and metadata."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    meta = {
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": label2id,
        "accuracy": accuracy,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "quick": args.quick,
    }
    with open(model_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Evaluate saved model
# ---------------------------------------------------------------------------

def evaluate_saved(args):
    """Load a saved model and evaluate on the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not MODEL_DIR.exists():
        print(f"No saved model at {MODEL_DIR}")
        sys.exit(1)

    # Load metadata for label mapping
    with open(MODEL_DIR / "training_metadata.json") as f:
        meta = json.load(f)
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    label2id = meta["label2id"]

    # Load data with same split
    texts_a, texts_b, labels_str = load_data(quick=False)
    classes = sorted(set(labels_str))
    labels_int = [label2id.get(l, label2id.get("other", 0)) for l in labels_str]

    _, test_a, _, test_b, _, test_labels = train_test_split(
        texts_a, texts_b, labels_int,
        test_size=0.20, random_state=42, stratify=labels_int,
    )
    print(f"Test set: {len(test_labels):,} clues")

    tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR))
    test_enc = tokenize(tokenizer, test_a, test_b, max_length=meta.get("max_length", 64))
    test_dataset = ClueDataset(test_enc, torch.tensor(test_labels, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)

    acc, report_str, cm = evaluate_model(model, test_loader, id2label, device)
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"\n{report_str}")
    print(f"\nConfusion matrix:\n{cm}")


# ---------------------------------------------------------------------------
# Predict single clue
# ---------------------------------------------------------------------------

def predict(args):
    """Predict wordplay type for a single clue."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_DIR.exists():
        print(f"No saved model at {MODEL_DIR}")
        sys.exit(1)

    with open(MODEL_DIR / "training_metadata.json") as f:
        meta = json.load(f)
    id2label = {int(k): v for k, v in meta["id2label"].items()}

    tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    clue_text = args.predict[0]
    answer = args.predict[1] if len(args.predict) > 1 else ""
    enum = args.predict[2] if len(args.predict) > 2 else ""

    text_a = clue_text
    text_b = f"{answer.upper()} ({enum})" if enum else answer.upper()

    enc = tokenizer(
        text_a, text_b,
        padding="max_length", truncation=True,
        max_length=meta.get("max_length", 64),
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()

    print(f"\nClue:   {clue_text}")
    print(f"Answer: {answer.upper()} ({enum})")
    print(f"\nPredicted: {id2label[pred_id]}  ({probs[pred_id]:.1%})")
    print(f"\nAll probabilities:")
    ranked = sorted(enumerate(probs.cpu().numpy()), key=lambda x: -x[1])
    for idx, prob in ranked:
        bar = "#" * int(prob * 40)
        print(f"  {id2label[idx]:20s} {prob:6.1%}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DB_PATH, MODEL_DIR

    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for clue type classification")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Train on 10%% of data")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model")
    parser.add_argument("--predict", nargs="+", metavar=("CLUE", "ANSWER"),
                        help="Predict: --predict 'clue text' ANSWER 7")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Path to database file")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR),
                        help="Directory to save/load model")

    args = parser.parse_args()
    DB_PATH = Path(args.db)
    MODEL_DIR = Path(args.model_dir)

    if args.predict:
        predict(args)
    elif args.evaluate:
        evaluate_saved(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
