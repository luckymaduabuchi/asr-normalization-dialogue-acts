#!/usr/bin/env python3
"""
Orchestrates train_transformer_worker.py across the 9 normalization conditions.

Each attempt runs the worker in a FRESH subprocess (not a retry loop inside one
long-lived process): a CUDA driver error on this shared vGPU appears to
permanently wedge the process's CUDA context, so retrying in-process kept
failing even with ample free memory. A new process gets a new CUDA context and
a real chance to succeed.

Results are appended to results_transformer/metrics_summary.csv incrementally,
and completed conditions (status OK) are skipped on resume.
"""
import os
import sys
import subprocess
import time
import pandas as pd

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "roberta-base"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results_transformer"
DATASET = sys.argv[3] if len(sys.argv) > 3 else "switchboard"
# MODEL_NAME == "bilstm" switches to the from-scratch BiLSTM worker, which has
# no pretrained checkpoint name to pass through.
WORKER_SCRIPT = "train_bilstm_worker.py" if MODEL_NAME == "bilstm" else "train_transformer_worker.py"
MAX_RETRIES = 5
RETRY_BACKOFF_S = 30

FULL_SWEEP = [
    "gold_text", "gold_alpha_text", "rm_punct_only", "rm_fillers_only",
    "rm_discourse_only", "norm_colloq_only", "clean_light", "clean_heavy",
    "asr_noise",
]
CORE_ONLY = ["gold_text", "rm_fillers_only", "clean_heavy"]

# All three corpora now have the full 9-condition ablation columns
# (extend_conditions_mrda_hvb.py added the 6 extra MRDA/HVB columns).
CONDITIONS = FULL_SWEEP


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
    # drop stale FAILED rows so a later success writes cleanly
    if (prev["status"] == "FAILED").any():
        prev[prev["status"] != "FAILED"].to_csv(path, index=False)
    return done


def run_condition(cond):
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n=== [{cond}] subprocess attempt {attempt}/{MAX_RETRIES} ===", flush=True)
        t0 = time.time()
        if WORKER_SCRIPT == "train_bilstm_worker.py":
            worker_args = [cond, OUT_DIR, DATASET]
        else:
            worker_args = [cond, MODEL_NAME, OUT_DIR, DATASET]
        proc = subprocess.run(
            [sys.executable, "-u", WORKER_SCRIPT] + worker_args,
            capture_output=False,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"[{cond}] subprocess succeeded in {elapsed/60:.1f} min", flush=True)
            return True
        print(f"[{cond}] subprocess exited {proc.returncode} after {elapsed/60:.1f} min, "
              f"retrying in {RETRY_BACKOFF_S}s", flush=True)
        time.sleep(RETRY_BACKOFF_S)
    return False


def main():
    done = load_done()
    print("Already completed:", done, flush=True)

    for cond in CONDITIONS:
        if cond in done:
            print(f"[skip] {cond} already completed", flush=True)
            continue

        ok = run_condition(cond)
        if not ok:
            print(f"[{cond}] giving up after {MAX_RETRIES} subprocess attempts", flush=True)
            append_metrics_row({"condition": cond, "macro_f1": None, "micro_f1": None, "accuracy": None, "status": "FAILED"})
            continue

        # worker already saved preds_<cond>.npy; recompute metrics from the RESULT_OK line
        # by re-reading the worker's own printed metrics via a small wrapper file it wrote.
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        y_test = np.load(os.path.join(OUT_DIR, "y_test_ids.npy"))
        preds = np.load(os.path.join(OUT_DIR, f"preds_{cond}.npy"))
        macro = f1_score(y_test, preds, average="macro", zero_division=0)
        micro = f1_score(y_test, preds, average="micro", zero_division=0)
        acc = accuracy_score(y_test, preds)
        append_metrics_row({"condition": cond, "macro_f1": float(macro), "micro_f1": float(micro), "accuracy": float(acc), "status": "OK"})
        print(f"{cond:18s} Macro-F1={macro:.4f}  Micro-F1={micro:.4f}  Acc={acc:.4f}", flush=True)

    print("\nAll conditions processed.", flush=True)


if __name__ == "__main__":
    main()
