#!/usr/bin/env python3
"""
Same retry/subprocess pattern as train_transformer_orchestrator.py, but for
an arbitrary, explicit list of conditions (used for the new
rm_fillers_strict / clean_heavy_strict / pragmatic_norm conditions rather
than the fixed 9-condition FULL_SWEEP). Appends new condition rows to the
SAME existing results_*/metrics_summary.csv used by the main grid (safe:
keyed by condition name, does not touch existing 9-condition rows).

Usage: python3 train_extra_conditions_orchestrator.py <model_name> <out_dir> <dataset> <cond1,cond2,...>
model_name == "bilstm" routes to train_bilstm_worker.py, same as the original orchestrator.
"""
import os
import sys
import subprocess
import time
import pandas as pd

MODEL_NAME = sys.argv[1]
OUT_DIR = sys.argv[2]
DATASET = sys.argv[3]
CONDITIONS = sys.argv[4].split(",")
WORKER_SCRIPT = "train_bilstm_worker.py" if MODEL_NAME == "bilstm" else "train_transformer_worker.py"
MAX_RETRIES = 5
RETRY_BACKOFF_S = 30


def append_metrics_row(row):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "metrics_summary.csv")
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def load_done():
    path = os.path.join(OUT_DIR, "metrics_summary.csv")
    if not os.path.exists(path):
        return set()
    prev = pd.read_csv(path)
    done = set(prev[prev["status"] == "OK"]["condition"].tolist())
    if (prev["status"] == "FAILED").any():
        prev[prev["status"] != "FAILED"].to_csv(path, index=False)
    return done


def run_condition(cond):
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n=== [{DATASET}/{MODEL_NAME}/{cond}] subprocess attempt {attempt}/{MAX_RETRIES} ===", flush=True)
        t0 = time.time()
        if WORKER_SCRIPT == "train_bilstm_worker.py":
            worker_args = [cond, OUT_DIR, DATASET]
        else:
            worker_args = [cond, MODEL_NAME, OUT_DIR, DATASET]
        proc = subprocess.run([sys.executable, "-u", WORKER_SCRIPT] + worker_args, capture_output=False)
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"[{cond}] subprocess succeeded in {elapsed/60:.1f} min", flush=True)
            return True
        print(f"[{cond}] subprocess exited {proc.returncode} after {elapsed/60:.1f} min, retrying in {RETRY_BACKOFF_S}s", flush=True)
        time.sleep(RETRY_BACKOFF_S)
    return False


def main():
    done = load_done()
    print(f"[{DATASET}/{MODEL_NAME}] Already completed: {done}", flush=True)

    for cond in CONDITIONS:
        if cond in done:
            print(f"[skip] {cond} already completed", flush=True)
            continue
        ok = run_condition(cond)
        if not ok:
            append_metrics_row({"condition": cond, "macro_f1": None, "micro_f1": None, "accuracy": None, "status": "FAILED"})
            continue

        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        y_test = np.load(os.path.join(OUT_DIR, "y_test_ids.npy"))
        preds = np.load(os.path.join(OUT_DIR, f"preds_{cond}.npy"))
        macro = f1_score(y_test, preds, average="macro", zero_division=0)
        micro = f1_score(y_test, preds, average="micro", zero_division=0)
        acc = accuracy_score(y_test, preds)
        append_metrics_row({"condition": cond, "macro_f1": float(macro), "micro_f1": float(micro), "accuracy": float(acc), "status": "OK"})
        print(f"{cond:20s} Macro-F1={macro:.4f}  Micro-F1={micro:.4f}  Acc={acc:.4f}", flush=True)

    print(f"\n[{DATASET}/{MODEL_NAME}] All conditions processed.", flush=True)


if __name__ == "__main__":
    main()
