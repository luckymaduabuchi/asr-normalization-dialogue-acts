#!/usr/bin/env python3
"""
Worker for the meeting-disjoint MRDA real-ASR sub-experiment (replaces the
exact-text-deduplication workaround in scripts/real_asr/mrda_real_asr_worker.py). Trains
EACH model from scratch on the train-meeting partition of
results/legacy/mrda_real/scoped_utterances.csv (60 of 75 ICSI meetings,
results/mrda/meetingsplit/split_meetings.json) and evaluates on the
held-out 15 test meetings' gold_text and Whisper asr_real transcripts.
Because both train and test come from the same ICSI-NXT-XML-derived
dataset with an explicit meeting column, disjointness is verified directly
rather than inferred from exact-text matching against a training split
that carries no meeting identifier.

Usage: python3 scripts/real_asr/mrda_meetingdisjoint_worker.py <model_key> [out_dir]
model_key in {tfidf, roberta-base, bert-base-cased, albert-base-v2, gpt2, bilstm}
Writes <out_dir>/preds_<model_key>_{gold,asr}.npy, y_test_ids.npy, label2id.json.
Prints "RESULT_OK <gold_macro> <gold_micro> <gold_acc> <asr_macro> <asr_micro> <asr_acc>".
"""
import os
import sys
import json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score

from mrda_real_asr_worker import run_tfidf, run_transformer, run_bilstm, MIN_COUNT

SCOPED_CSV = "results/legacy/mrda_real/scoped_utterances.csv"
ASR_CSV = "results/legacy/mrda_real/asr_real.csv"
SPLIT_JSON = "results/mrda/meetingsplit/split_meetings.json"


def load_data():
    scoped = pd.read_csv(SCOPED_CSV).rename(columns={"general_da": "label"})
    asr = pd.read_csv(ASR_CSV)
    split = json.load(open(SPLIT_JSON))
    train_meetings = set(split["train_meetings"])
    test_meetings = set(split["test_meetings"])

    vc = scoped["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    scoped = scoped[scoped["label"].isin(keep)].copy()

    train_df = scoped[scoped["meeting"].isin(train_meetings)].copy()
    test_df = scoped[scoped["meeting"].isin(test_meetings)].copy()
    assert set(train_df["meeting"]).isdisjoint(set(test_df["meeting"]))

    test_df = test_df.merge(asr, on="utterance_id", how="inner")
    test_df["asr_real"] = test_df["asr_real"].fillna("")
    return train_df, test_df


def metrics(y_true, y_pred, labels_order):
    macro = f1_score(y_true, y_pred, average="macro", labels=list(range(len(labels_order))), zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", labels=list(range(len(labels_order))), zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return macro, micro, acc


def main():
    model_key = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/mrda/meetingsplit"
    os.makedirs(out_dir, exist_ok=True)

    train_df, test_df = load_data()
    labels_order = sorted(train_df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels_order)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_key}] train={len(train_df)} test={len(test_df)} "
          f"train_meetings={train_df['meeting'].nunique()} test_meetings={test_df['meeting'].nunique()} "
          f"labels={labels_order} device={device}", flush=True)

    if model_key == "tfidf":
        y_test, pred_gold, pred_asr = run_tfidf(train_df, test_df, label2id, out_dir)
    elif model_key == "bilstm":
        y_test, pred_gold, pred_asr = run_bilstm(train_df, test_df, label2id, device)
    else:
        y_test, pred_gold, pred_asr = run_transformer(model_key, train_df, test_df, label2id, device)

    gold_macro, gold_micro, gold_acc = metrics(y_test, pred_gold, labels_order)
    asr_macro, asr_micro, asr_acc = metrics(y_test, pred_asr, labels_order)

    np.save(os.path.join(out_dir, f"preds_{model_key}_gold.npy"), pred_gold)
    np.save(os.path.join(out_dir, f"preds_{model_key}_asr.npy"), pred_asr)
    np.save(os.path.join(out_dir, "y_test_ids.npy"), y_test)
    with open(os.path.join(out_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print(f"RESULT_OK {gold_macro:.6f} {gold_micro:.6f} {gold_acc:.6f} {asr_macro:.6f} {asr_micro:.6f} {asr_acc:.6f}", flush=True)


if __name__ == "__main__":
    main()
