#!/usr/bin/env python3
"""
Runs gold_text / rm_fillers_only / clean_heavy on Switchboard for 4 additional
seeds (1, 2, 3, 4; seed 42 is the existing main-grid result) across the 5
non-TF-IDF models, to report mean +/- sd and check whether the small
Switchboard-specific RoBERTa/GPT-2 recovery effect (and the other results)
are robust to seed variance rather than an artifact of a single run.

Writes to results/switchboard/multiseed/<model>_seed<k>/ (mirrors the per-condition
preds_<cond>.npy / metrics_summary.csv layout used elsewhere) and a combined
results/switchboard/multiseed/all_seeds_summary.csv at the end.
"""
import os
import sys
import subprocess
import time
import pandas as pd

MODELS = ["roberta-base", "bert-base-cased", "albert-base-v2", "gpt2", "bilstm"]
CONDITIONS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["gold_text", "rm_fillers_only", "clean_heavy"]
SEEDS = [1, 2, 3, 4]
DATASET = sys.argv[1] if len(sys.argv) > 1 else "switchboard"
MAX_RETRIES = 5
RETRY_BACKOFF_S = 30
BASE_OUT = sys.argv[2] if len(sys.argv) > 2 else "results/switchboard/multiseed"


def worker_script(model):
    return "scripts/train/train_bilstm_worker.py" if model == "bilstm" else "scripts/train/train_transformer_worker.py"


def worker_args(model, cond, out_dir, seed):
    if model == "bilstm":
        return [cond, out_dir, DATASET, str(seed)]
    return [cond, model, out_dir, DATASET, str(seed)]


def append_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def load_done(path):
    if not os.path.exists(path):
        return set()
    prev = pd.read_csv(path)
    done = set(prev[prev["status"] == "OK"]["condition"].tolist())
    if (prev["status"] == "FAILED").any():
        prev[prev["status"] != "FAILED"].to_csv(path, index=False)
    return done


def run_once(model, cond, out_dir, seed):
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n=== [{model}/seed{seed}/{cond}] attempt {attempt}/{MAX_RETRIES} ===", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-u", worker_script(model)] + worker_args(model, cond, out_dir, seed))
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"[{model}/seed{seed}/{cond}] succeeded in {elapsed/60:.1f} min", flush=True)
            return True
        print(f"[{model}/seed{seed}/{cond}] exited {proc.returncode} after {elapsed/60:.1f} min, retrying in {RETRY_BACKOFF_S}s", flush=True)
        time.sleep(RETRY_BACKOFF_S)
    return False


def main():
    all_rows = []
    for model in MODELS:
        for seed in SEEDS:
            out_dir = os.path.join(BASE_OUT, f"{model}_seed{seed}")
            metrics_path = os.path.join(out_dir, "metrics_summary.csv")
            done = load_done(metrics_path)
            for cond in CONDITIONS:
                if cond in done:
                    print(f"[skip] {model}/seed{seed}/{cond} already completed", flush=True)
                    continue
                ok = run_once(model, cond, out_dir, seed)
                if not ok:
                    append_row(metrics_path, {"condition": cond, "macro_f1": None, "micro_f1": None, "accuracy": None, "status": "FAILED"})
                    continue
                import numpy as np
                from sklearn.metrics import f1_score, accuracy_score
                y_test = np.load(os.path.join(out_dir, "y_test_ids.npy"))
                preds = np.load(os.path.join(out_dir, f"preds_{cond}.npy"))
                macro = f1_score(y_test, preds, average="macro", zero_division=0)
                micro = f1_score(y_test, preds, average="micro", zero_division=0)
                acc = accuracy_score(y_test, preds)
                append_row(metrics_path, {"condition": cond, "macro_f1": float(macro), "micro_f1": float(micro), "accuracy": float(acc), "status": "OK"})
                print(f"{model}/seed{seed}/{cond:16s} Macro-F1={macro:.4f}", flush=True)

    # Aggregate across all seed dirs (including any completed in earlier resumed runs)
    for model in MODELS:
        for seed in SEEDS:
            out_dir = os.path.join(BASE_OUT, f"{model}_seed{seed}")
            metrics_path = os.path.join(out_dir, "metrics_summary.csv")
            if not os.path.exists(metrics_path):
                continue
            df = pd.read_csv(metrics_path)
            df = df[df["status"] == "OK"].copy()
            df["model"] = model
            df["seed"] = seed
            all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(os.path.join(BASE_OUT, "all_seeds_summary.csv"), index=False)
        print("\nSaved combined summary to results/switchboard/multiseed/all_seeds_summary.csv")

    print("\nAll multi-seed runs processed.", flush=True)


if __name__ == "__main__":
    main()
