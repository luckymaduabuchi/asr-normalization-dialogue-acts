#!/usr/bin/env python3
"""
Trains rm_fillers_only and clean_heavy (seed 42 only) for all 5 non-TF-IDF
models on the meeting-disjoint MRDA split, so that a meeting-clustered
permutation/bootstrap significance test (matching what TF-IDF+LR already
gets) becomes possible for the neural models too. gold_text is not retrained
here -- it's already trained per model in results/mrda/<model>/ from the
real-ASR experiment and reused directly.
"""
import os
import sys
import subprocess
import time
import pandas as pd

MODELS = ["roberta-base", "bert-base-cased", "albert-base-v2", "gpt2", "bilstm"]
CONDITIONS = ["rm_fillers_only", "clean_heavy"]
DATASET = "mrda_meetingdisjoint"
MAX_RETRIES = 5
RETRY_BACKOFF_S = 30

OUT_DIRS = {
    "roberta-base": "results/mrda/roberta",
    "bert-base-cased": "results/mrda/bert",
    "albert-base-v2": "results/mrda/albert",
    "gpt2": "results/mrda/gpt2",
    "bilstm": "results/mrda/bilstm",
}


def worker_script(model):
    return "scripts/train/train_bilstm_worker.py" if model == "bilstm" else "scripts/train/train_transformer_worker.py"


def worker_args(model, cond, out_dir):
    if model == "bilstm":
        return [cond, out_dir, DATASET]
    return [cond, model, out_dir, DATASET]


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
    return set(prev[prev["status"] == "OK"]["condition"].tolist())


def main():
    for model in MODELS:
        out_dir = os.path.join("results/mrda/meetingsplit_conditions", model)
        metrics_path = os.path.join(out_dir, "metrics_summary.csv")
        done = load_done(metrics_path)
        for cond in CONDITIONS:
            if cond in done:
                print(f"[skip] {model}/{cond} already completed", flush=True)
                continue
            ok = False
            for attempt in range(1, MAX_RETRIES + 1):
                print(f"\n=== [{model}/{cond}] attempt {attempt}/{MAX_RETRIES} ===", flush=True)
                t0 = time.time()
                proc = subprocess.run([sys.executable, "-u", worker_script(model)] + worker_args(model, cond, out_dir))
                elapsed = time.time() - t0
                if proc.returncode == 0:
                    print(f"[{model}/{cond}] succeeded in {elapsed/60:.1f} min", flush=True)
                    ok = True
                    break
                print(f"[{model}/{cond}] exited {proc.returncode} after {elapsed/60:.1f} min, retrying in {RETRY_BACKOFF_S}s", flush=True)
                time.sleep(RETRY_BACKOFF_S)
            if not ok:
                append_row(metrics_path, {"condition": cond, "macro_f1": None, "micro_f1": None, "accuracy": None, "status": "FAILED"})
                continue
            import numpy as np
            from sklearn.metrics import f1_score, accuracy_score
            y_test = np.load(os.path.join(out_dir, "y_test_ids.npy"))
            preds = np.load(os.path.join(out_dir, f"preds_{cond}.npy"))
            macro = f1_score(y_test, preds, average="macro", zero_division=0)
            acc = accuracy_score(y_test, preds)
            append_row(metrics_path, {"condition": cond, "macro_f1": float(macro), "micro_f1": float(acc), "accuracy": float(acc), "status": "OK"})
            print(f"{model}/{cond}: Macro-F1={macro:.4f}", flush=True)

    print("\nAll MRDA meeting-disjoint condition runs processed.", flush=True)


if __name__ == "__main__":
    main()
