#!/usr/bin/env python3
"""
SUPERSEDED -- do not run. Use scripts/train/train_transformer_orchestrator.py (which calls
scripts/train/train_transformer_worker.py as a subprocess per condition) instead.

This single-process version was built while misdiagnosing a "CUDA driver error:
operation not supported" failure as GPU memory contention with other tenants on
this shared vGPU. The batch=2/grad-accum=8/gradient-checkpointing/Adafactor
settings below, and the whole retry-with-backoff design, were compensating for
that misdiagnosis. The actual cause turned out to be unrelated to memory or
contention: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set below) enables
a CUDA allocator mode that uses low-level virtual-memory APIs this vGPU profile
doesn't support, and failed deterministically once the allocator needed to grow
its pool. With that env var removed, plain model.to('cuda') succeeds immediately
and consistently -- see scripts/train/train_transformer_worker.py's docstring for the full
diagnosis. No retries, subprocess isolation, or reduced batch size are needed.

Original docstring follows:

Fine-tunes roberta-base separately for each normalization condition, reusing the
exact conversation-level train/test split saved by scripts/train/train_ablations.py
(results/switchboard/tfidf/split_conversations.json) so results are directly comparable to
Table tab:ablations (TF-IDF+LR). This is the contextual-model generalizability
check requested by Reviewer #1.

Uses a class-balanced weighted cross-entropy loss (matching class_weight="balanced"
used for the LR baseline) via a Trainer subclass.
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, accuracy_score

DATA = "utterances_with_ablations.csv"
SPLIT_FILE = "results/switchboard/tfidf/split_conversations.json"
OUT_DIR = "results/switchboard/roberta"
MIN_COUNT = 50
MODEL_NAME = "roberta-base"
MAX_LENGTH = 96
SEED = 42
MAX_RETRIES = 8
MIN_FREE_MB = 1800
PREFLIGHT_MAX_WAIT_S = 1800

CONDITIONS = [
    "gold_text", "gold_alpha_text", "rm_punct_only", "rm_fillers_only",
    "rm_discourse_only", "norm_colloq_only", "clean_light", "clean_heavy",
    "asr_noise",
]

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def free_gpu_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


def wait_for_gpu_headroom(min_free_mb=MIN_FREE_MB, max_wait_s=PREFLIGHT_MAX_WAIT_S):
    """This vGPU is shared with 2-3 unrelated heavy tenants (other users' jobs in
    other conda envs) whose memory usage swings widely and unpredictably. Rather
    than launching straight into a contended window and hitting a CUDA error,
    poll free memory first and wait for at least min_free_mb before proceeding."""
    waited = 0
    interval = 15
    while waited < max_wait_s:
        free_mb = free_gpu_mb()
        if free_mb is None or free_mb >= min_free_mb:
            return free_mb
        time.sleep(interval)
        waited += interval
    return free_gpu_mb()


def append_metrics_row(row):
    path = os.path.join(OUT_DIR, "metrics_summary.csv")
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def run_condition_with_retries(cond, train_ds, test_ds, label2id, id2label, class_weights):
    for attempt in range(1, MAX_RETRIES + 1):
        free_mb = wait_for_gpu_headroom()
        print(f"[{cond}] attempt {attempt}/{MAX_RETRIES}, free GPU mem ~{free_mb} MiB")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=len(label2id), label2id=label2id, id2label=id2label
            )
            model.gradient_checkpointing_enable()

            args = TrainingArguments(
                output_dir=os.path.join(OUT_DIR, f"tmp_{cond}"),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                learning_rate=2e-5,
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),
                optim="adafactor",
                logging_steps=200,
                save_strategy="no",
                eval_strategy="no",
                report_to=[],
                seed=SEED,
                disable_tqdm=False,
            )

            trainer = WeightedTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                class_weights=class_weights,
            )
            trainer.train()

            pred_logits = trainer.predict(test_ds).predictions
            pred_ids = np.argmax(pred_logits, axis=1)

            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

            return pred_ids

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"[{cond}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            try:
                del model
            except NameError:
                pass
            try:
                del trainer
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(min(60 * attempt, 300))
    print(f"[{cond}] giving up after {MAX_RETRIES} attempts")
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    df = pd.read_csv(DATA).dropna(subset=["label"]).copy()
    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    df = df[df["label"].isin(keep)].copy()
    df["conversation_id"] = df["conversation_id"].astype(str)

    split = json.load(open(SPLIT_FILE))
    train_convs = set(split["train_conversation_ids"])
    test_convs = set(split["test_conversation_ids"])

    train_df = df[df["conversation_id"].isin(train_convs)].copy()
    test_df = df[df["conversation_id"].isin(test_convs)].copy()

    labels_order = sorted(train_df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels_order)}
    id2label = {i: l for l, i in label2id.items()}

    y_train_ids = train_df["label"].map(label2id).to_numpy()
    y_test_ids = test_df["label"].map(label2id).to_numpy()

    counts = np.bincount(y_train_ids, minlength=len(labels_order)).astype(float)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(
        len(y_train_ids) / (len(labels_order) * counts), dtype=torch.float
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Resume support: skip conditions already completed in a previous partial run
    done_conditions = set()
    existing_summary = os.path.join(OUT_DIR, "metrics_summary.csv")
    if os.path.exists(existing_summary):
        prev = pd.read_csv(existing_summary)
        done_conditions = set(prev[prev["status"] == "OK"]["condition"].tolist())
        failed_conditions = set(prev[prev["status"] == "FAILED"]["condition"].tolist())
        if failed_conditions:
            # drop stale FAILED rows for conditions we're about to retry; the
            # successful retry's row will be appended fresh below
            prev = prev[prev["status"] != "FAILED"]
            prev.to_csv(existing_summary, index=False)
        print("Already completed (OK):", done_conditions)
        print("Previously failed, will retry:", failed_conditions)

    np.save(os.path.join(OUT_DIR, "y_test_ids.npy"), y_test_ids)
    with open(os.path.join(OUT_DIR, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    for cond in CONDITIONS:
        if cond not in df.columns:
            print(f"[skip] column {cond} missing")
            continue
        if cond in done_conditions:
            print(f"[skip] {cond} already completed")
            continue

        print(f"\n=== Fine-tuning roberta-base on condition: {cond} ===")
        t0 = time.time()

        train_texts = train_df[cond].fillna("").tolist()
        test_texts = test_df[cond].fillna("").tolist()

        train_ds = Dataset.from_dict({"text": train_texts, "labels": y_train_ids.tolist()})
        test_ds = Dataset.from_dict({"text": test_texts, "labels": y_test_ids.tolist()})

        train_ds = train_ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"), batched=True)
        test_ds = test_ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"), batched=True)
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        pred_ids = run_condition_with_retries(cond, train_ds, test_ds, label2id, id2label, class_weights)

        if pred_ids is None:
            append_metrics_row({"condition": cond, "macro_f1": None, "micro_f1": None, "accuracy": None, "status": "FAILED"})
            continue

        macro = f1_score(y_test_ids, pred_ids, average="macro", zero_division=0)
        micro = f1_score(y_test_ids, pred_ids, average="micro", zero_division=0)
        acc = accuracy_score(y_test_ids, pred_ids)
        elapsed = time.time() - t0

        print(f"{cond:18s} Macro-F1={macro:.4f}  Micro-F1={micro:.4f}  Acc={acc:.4f}  ({elapsed/60:.1f} min)")

        np.save(os.path.join(OUT_DIR, f"preds_{cond}.npy"), pred_ids)
        append_metrics_row({
            "condition": cond, "macro_f1": float(macro), "micro_f1": float(micro),
            "accuracy": float(acc), "status": "OK", "minutes": round(elapsed / 60, 1),
        })

    print("\nAll conditions processed. Saved to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
