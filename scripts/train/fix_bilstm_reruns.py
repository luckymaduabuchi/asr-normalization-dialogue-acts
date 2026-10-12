#!/usr/bin/env python3
"""
Reruns every BiLSTM (dataset, condition[, seed]) cell that was computed with
the buggy masked-max-pool (see scripts/train/train_bilstm_worker.py fix: an all-padding
row -- produced whenever a condition strips an utterance down to nothing,
e.g. a filler-only HVB/MRDA utterance under rm_fillers_only -- pushed a
-1e9 sentinel through the classifier's Linear layer, exploding logits to
~1e8-1e9 and corrupting gradient updates for every batch containing such a
row). Affected fraction of training rows: Switchboard clean_heavy 1.1%,
MRDA rm_fillers_only/clean_heavy ~9.0%, MRDA asr_noise 2.7%, MRDA strict
variants 2.4%, HVB rm_fillers_only/clean_heavy/asr_noise ~0.8-1.0%, HVB
strict variants 0.4%. gold_text and any condition with 0 empty rows
(verified separately) needs no rerun.

Removes the stale row for each target condition from its metrics_summary.csv
(so the orchestrator-style skip-if-done logic doesn't apply) then reruns
scripts/train/train_bilstm_worker.py directly and appends the corrected row.
"""
import os
import sys
import subprocess
import time
import pandas as pd

MAX_RETRIES = 5
RETRY_BACKOFF_S = 30

# (out_dir, dataset, condition, seed_or_None)
TARGETS = [
    # --- Main grid (seed 42) ---
    ("results/switchboard/bilstm", "switchboard", "clean_heavy", None),
    ("results/mrda/bilstm", "mrda", "rm_fillers_only", None),
    ("results/mrda/bilstm", "mrda", "clean_heavy", None),
    ("results/mrda/bilstm", "mrda", "asr_noise", None),
    ("results/hvb/bilstm", "hvb", "rm_fillers_only", None),
    ("results/hvb/bilstm", "hvb", "clean_heavy", None),
    ("results/hvb/bilstm", "hvb", "asr_noise", None),
    # --- Strict-filler robustness check (seed 42) ---
    ("results/switchboard/bilstm", "switchboard", "rm_fillers_strict", None),
    ("results/switchboard/bilstm", "switchboard", "clean_heavy_strict", None),
    ("results/mrda/bilstm", "mrda", "rm_fillers_strict", None),
    ("results/mrda/bilstm", "mrda", "clean_heavy_strict", None),
    ("results/hvb/bilstm", "hvb", "rm_fillers_strict", None),
    ("results/hvb/bilstm", "hvb", "clean_heavy_strict", None),
    # --- Switchboard multi-seed (clean_heavy only; rm_fillers_only/gold_text unaffected) ---
    ("results/switchboard/multiseed/bilstm_seed1", "switchboard", "clean_heavy", 1),
    ("results/switchboard/multiseed/bilstm_seed2", "switchboard", "clean_heavy", 2),
    ("results/switchboard/multiseed/bilstm_seed3", "switchboard", "clean_heavy", 3),
    ("results/switchboard/multiseed/bilstm_seed4", "switchboard", "clean_heavy", 4),
    # --- HVB multi-seed (rm_fillers_only + clean_heavy) ---
    ("results/hvb/multiseed/bilstm_seed1", "hvb", "rm_fillers_only", 1),
    ("results/hvb/multiseed/bilstm_seed1", "hvb", "clean_heavy", 1),
    ("results/hvb/multiseed/bilstm_seed2", "hvb", "rm_fillers_only", 2),
    ("results/hvb/multiseed/bilstm_seed2", "hvb", "clean_heavy", 2),
    ("results/hvb/multiseed/bilstm_seed3", "hvb", "rm_fillers_only", 3),
    ("results/hvb/multiseed/bilstm_seed3", "hvb", "clean_heavy", 3),
    ("results/hvb/multiseed/bilstm_seed4", "hvb", "rm_fillers_only", 4),
    ("results/hvb/multiseed/bilstm_seed4", "hvb", "clean_heavy", 4),
]


def remove_stale_row(out_dir, cond):
    path = os.path.join(out_dir, "metrics_summary.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["condition"] != cond]
    if len(df) != before:
        df.to_csv(path, index=False)
        print(f"  removed stale row: {path} [{cond}]")


def append_row(out_dir, row):
    path = os.path.join(out_dir, "metrics_summary.csv")
    os.makedirs(out_dir, exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def main():
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score

    for out_dir, dataset, cond, seed in TARGETS:
        print(f"\n=== [{dataset}/{cond}/seed={seed or 42}] -> {out_dir} ===", flush=True)
        remove_stale_row(out_dir, cond)
        args = [cond, out_dir, dataset] + ([str(seed)] if seed is not None else [])
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            proc = subprocess.run([sys.executable, "-u", "scripts/train/train_bilstm_worker.py"] + args)
            if proc.returncode == 0:
                ok = True
                break
            print(f"  attempt {attempt}/{MAX_RETRIES} exited {proc.returncode}, retrying in {RETRY_BACKOFF_S}s", flush=True)
            time.sleep(RETRY_BACKOFF_S)
        if not ok:
            print(f"  FAILED: {dataset}/{cond}/seed={seed}", flush=True)
            append_row(out_dir, {"condition": cond, "macro_f1": None, "micro_f1": None, "accuracy": None, "status": "FAILED"})
            continue
        y_test = np.load(os.path.join(out_dir, "y_test_ids.npy"))
        preds = np.load(os.path.join(out_dir, f"preds_{cond}.npy"))
        macro = f1_score(y_test, preds, average="macro", zero_division=0)
        micro = f1_score(y_test, preds, average="micro", zero_division=0)
        acc = accuracy_score(y_test, preds)
        append_row(out_dir, {"condition": cond, "macro_f1": float(macro), "micro_f1": float(micro), "accuracy": float(acc), "status": "OK"})
        print(f"  {dataset}/{cond}/seed={seed or 42}: Macro-F1={macro:.4f} (fixed)", flush=True)

    print("\nAll BiLSTM fix-reruns complete.", flush=True)


if __name__ == "__main__":
    main()
