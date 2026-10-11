#!/usr/bin/env python3
"""
Runs mrda_meetingdisjoint_worker.py once per model (fresh subprocess each,
matching the retry pattern used for the earlier scoped MRDA real-ASR run --
this shared GPU intermittently OOMs/wedges under contention from other
processes). Resumable: skips models already marked OK in metrics_summary.csv.
"""
import os
import sys
import subprocess
import time
import json
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

OUT_DIR = "results_mrda_meetingsplit"
MODELS = [
    ("TF-IDF+LR", "tfidf"),
    ("RoBERTa", "roberta-base"),
    ("BERT", "bert-base-cased"),
    ("ALBERT", "albert-base-v2"),
    ("GPT-2", "gpt2"),
    ("BiLSTM", "bilstm"),
]
MAX_RETRIES = 5
RETRY_BACKOFF_S = 30


def load_done():
    path = os.path.join(OUT_DIR, "metrics_summary.csv")
    if not os.path.exists(path):
        return set()
    prev = pd.read_csv(path)
    done = set(prev[prev["status"] == "OK"]["model"].tolist())
    if (prev["status"] == "FAILED").any():
        prev[prev["status"] != "FAILED"].to_csv(path, index=False)
    return done


def append_row(row):
    path = os.path.join(OUT_DIR, "metrics_summary.csv")
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        df_row.to_csv(path, mode="w", header=True, index=False)


def run_model(model_key):
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n=== [{model_key}] subprocess attempt {attempt}/{MAX_RETRIES} ===", flush=True)
        t0 = time.time()
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128")
        proc = subprocess.run([sys.executable, "-u", "mrda_meetingdisjoint_worker.py", model_key, OUT_DIR], env=env)
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"[{model_key}] succeeded in {elapsed/60:.1f} min", flush=True)
            return True
        print(f"[{model_key}] exited {proc.returncode} after {elapsed/60:.1f} min, retrying in {RETRY_BACKOFF_S}s", flush=True)
        time.sleep(RETRY_BACKOFF_S)
    return False


def main():
    done = load_done()
    print("Already completed:", done, flush=True)

    mcnemar_rows = []
    for model_name, model_key in MODELS:
        if model_name in done:
            print(f"[skip] {model_name} already completed", flush=True)
            continue
        ok = run_model(model_key)
        if not ok:
            append_row({"model": model_name, "gold_macro_f1": None, "gold_micro_f1": None, "gold_acc": None,
                        "asr_real_macro_f1": None, "asr_real_micro_f1": None, "asr_real_acc": None, "status": "FAILED"})
            continue

        y_test = np.load(os.path.join(OUT_DIR, "y_test_ids.npy"))
        pred_gold = np.load(os.path.join(OUT_DIR, f"preds_{model_key}_gold.npy"))
        pred_asr = np.load(os.path.join(OUT_DIR, f"preds_{model_key}_asr.npy"))

        from sklearn.metrics import f1_score, accuracy_score
        n_labels = len(json.load(open(os.path.join(OUT_DIR, "label2id.json"))))
        gold_macro = f1_score(y_test, pred_gold, average="macro", labels=list(range(n_labels)), zero_division=0)
        gold_micro = f1_score(y_test, pred_gold, average="micro", labels=list(range(n_labels)), zero_division=0)
        gold_acc = accuracy_score(y_test, pred_gold)
        asr_macro = f1_score(y_test, pred_asr, average="macro", labels=list(range(n_labels)), zero_division=0)
        asr_micro = f1_score(y_test, pred_asr, average="micro", labels=list(range(n_labels)), zero_division=0)
        asr_acc = accuracy_score(y_test, pred_asr)

        append_row({"model": model_name, "gold_macro_f1": float(gold_macro), "gold_micro_f1": float(gold_micro),
                    "gold_acc": float(gold_acc), "asr_real_macro_f1": float(asr_macro),
                    "asr_real_micro_f1": float(asr_micro), "asr_real_acc": float(asr_acc), "status": "OK"})
        print(f"{model_name:12s} gold: macro={gold_macro:.4f} micro={gold_micro:.4f} acc={gold_acc:.4f}  |  "
              f"asr_real: macro={asr_macro:.4f} micro={asr_micro:.4f} acc={asr_acc:.4f}", flush=True)

        gold_correct = (pred_gold == y_test)
        asr_correct = (pred_asr == y_test)
        gold_only = int(np.sum(gold_correct & ~asr_correct))
        asr_only = int(np.sum(~gold_correct & asr_correct))
        both_correct = int(np.sum(gold_correct & asr_correct))
        both_wrong = int(np.sum(~gold_correct & ~asr_correct))
        table = [[both_correct, gold_only], [asr_only, both_wrong]]
        use_exact = (gold_only + asr_only) < 5000
        result = mcnemar(table, exact=use_exact, correction=not use_exact)
        mcnemar_rows.append({"model": model_name, "gold_only_correct": gold_only, "asr_real_only_correct": asr_only,
                              "statistic": float(result.statistic), "p_value": float(result.pvalue)})
        print(f"McNemar {model_name} gold vs asr_real: gold-only={gold_only}, asr-only={asr_only}, p={result.pvalue:.4g}", flush=True)

    if mcnemar_rows:
        mpath = os.path.join(OUT_DIR, "mcnemar_tests.csv")
        new_df = pd.DataFrame(mcnemar_rows)
        if os.path.exists(mpath):
            old_df = pd.read_csv(mpath)
            new_df = pd.concat([old_df[~old_df["model"].isin(new_df["model"])], new_df], ignore_index=True)
        new_df.to_csv(mpath, index=False)

    print("\nAll models processed.", flush=True)


if __name__ == "__main__":
    main()
