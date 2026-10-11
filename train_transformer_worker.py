#!/usr/bin/env python3
"""
Worker: fine-tunes one transformer model on exactly ONE normalization condition
and exits.

Root cause of the earlier "CUDA driver error: operation not supported" failures
(traced down after ruling out memory, contention timing, and CUDA build version):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True enables PyTorch's virtual-memory
based CUDA allocator, which uses low-level CUDA VMM APIs (cuMemCreate/cuMemMap)
that this vGPU profile (`nvidia-smi -q` reports Virtualization Mode: VGPU) does
not support. It worked for the first several dozen small allocations and then
failed hard and permanently once the allocator needed to grow its pool -- which
looked exactly like "transient contention" until tested directly with that env
var unset, at which point model loading succeeded instantly and repeatedly.
Fix: don't set expandable_segments. No retry/subprocess gymnastics needed.

Usage: python3 train_transformer_worker.py <condition> [model_name] [out_dir] [dataset]
model_name defaults to roberta-base, out_dir defaults to results_transformer,
dataset is one of switchboard (default) / mrda / hvb.
Writes <out_dir>/preds_<condition>.npy and prints "RESULT_OK <macro> <micro> <acc>"
on success; a nonzero exit / crash is treated as failure by the orchestrator.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, accuracy_score

MIN_COUNT = 50
MAX_LENGTH = 96
SEED = 42

# dataset registry: how to load train/test text+label frames for each corpus
DATASETS = {
    "switchboard": {
        "csv": "utterances_with_ablations.csv",
        "split_file": "results_ablation/split_conversations.json",
        "conv_col": "conversation_id",
    },
    "mrda": {
        "csv": "results_mrda/utterances_mrda.csv",
        "split_file": None,  # has its own "split" column
    },
    "hvb": {
        "csv": "results_hvb/utterances_hvb.csv",
        "split_file": None,  # has its own "split" column
    },
    "mrda_meetingdisjoint": {
        "csv": "results_mrda_meetingsplit/scoped_utterances_with_conditions.csv",
        "split_file": None,  # has its own "split" column (derived from split_meetings.json)
    },
}


def load_train_test(dataset_key):
    cfg = DATASETS[dataset_key]
    df = pd.read_csv(cfg["csv"]).dropna(subset=["label"]).copy()
    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    df = df[df["label"].isin(keep)].copy()

    if cfg["split_file"] is not None:
        df[cfg["conv_col"]] = df[cfg["conv_col"]].astype(str)
        split = json.load(open(cfg["split_file"]))
        train_convs = set(split["train_conversation_ids"])
        test_convs = set(split["test_conversation_ids"])
        train_df = df[df[cfg["conv_col"]].isin(train_convs)].copy()
        test_df = df[df[cfg["conv_col"]].isin(test_convs)].copy()
    else:
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()

    return train_df, test_df


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


def main():
    cond = sys.argv[1]
    MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "roberta-base"
    OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "results_transformer"
    DATASET = sys.argv[4] if len(sys.argv) > 4 else "switchboard"
    RUN_SEED = int(sys.argv[5]) if len(sys.argv) > 5 else SEED
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(RUN_SEED)

    train_df, test_df = load_train_test(DATASET)

    labels_order = sorted(train_df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels_order)}
    id2label = {i: l for l, i in label2id.items()}

    y_train_ids = train_df["label"].map(label2id).to_numpy()
    y_test_ids = test_df["label"].map(label2id).to_numpy()

    counts = np.bincount(y_train_ids, minlength=len(labels_order)).astype(float)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(len(y_train_ids) / (len(labels_order) * counts), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels_order), label2id=label2id, id2label=id2label
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        # GPT-2 (decoder-only) has no pad token by default; since it's causal,
        # classification pools from the last non-pad token position, so any
        # fixed pad token works as long as tokenizer and model agree on it.
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if torch.cuda.is_available():
        model = model.to("cuda")

    train_texts = train_df[cond].fillna("").tolist()
    test_texts = test_df[cond].fillna("").tolist()

    train_ds = Dataset.from_dict({"text": train_texts, "labels": y_train_ids.tolist()})
    test_ds = Dataset.from_dict({"text": test_texts, "labels": y_test_ids.tolist()})
    train_ds = train_ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"), batched=True)
    test_ds = test_ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"), batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, f"tmp_{cond}"),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=RUN_SEED,
        disable_tqdm=False,
    )

    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, class_weights=class_weights)
    trainer.train()

    pred_logits = trainer.predict(test_ds).predictions
    pred_ids = np.argmax(pred_logits, axis=1)

    macro = f1_score(y_test_ids, pred_ids, average="macro", zero_division=0)
    micro = f1_score(y_test_ids, pred_ids, average="micro", zero_division=0)
    acc = accuracy_score(y_test_ids, pred_ids)

    np.save(os.path.join(OUT_DIR, f"preds_{cond}.npy"), pred_ids)
    np.save(os.path.join(OUT_DIR, "y_test_ids.npy"), y_test_ids)
    with open(os.path.join(OUT_DIR, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print(f"RESULT_OK {macro:.6f} {micro:.6f} {acc:.6f}")


if __name__ == "__main__":
    main()
