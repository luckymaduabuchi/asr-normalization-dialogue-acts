#!/usr/bin/env python3
"""
Extends cluster-level (conversation-/issue-level) permutation testing and
bootstrap CIs from TF-IDF+LR (already reported) to the five neural models,
for Switchboard (conversation_id) and HVB (issue_id). Uses already-saved
prediction arrays -- no retraining. MRDA is handled separately (see
scripts/train/train_mrda_meetingdisjoint_conditions.py + a follow-up script), since its
main-grid split has no meeting identifier to cluster by.

For each corpus and model, loads y_test_ids.npy + preds_gold_text.npy +
preds_rm_fillers_only.npy + preds_clean_heavy.npy from that model's results
directory, re-derives the test dataframe's cluster-id column in the same
row order (via train_transformer_worker.load_train_test, which the worker
scripts used to build the same test set), and runs a cluster-level
permutation test (2000 iters) + bootstrap CI (2000 resamples) on the
Macro-F1 delta, exactly mirroring scripts/significance/significance_tests.py's Switchboard
TF-IDF+LR design.

Output: results/cross_corpus/significance_neural/<corpus>_<model>.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from train_transformer_worker import load_train_test, MIN_COUNT  # noqa: E402

OUT_DIR = "results/cross_corpus/significance_neural"
os.makedirs(OUT_DIR, exist_ok=True)

CORPUS_CFG = {
    "switchboard": {"cluster_col": "conversation_id", "models": {
        "RoBERTa": "results/switchboard/roberta", "BERT": "results/switchboard/bert",
        "ALBERT": "results/switchboard/albert", "GPT-2": "results/switchboard/gpt2",
        "BiLSTM": "results/switchboard/bilstm",
    }},
    "hvb": {"cluster_col": "issue_id", "models": {
        "RoBERTa": "results/hvb/roberta", "BERT": "results/hvb/bert",
        "ALBERT": "results/hvb/albert", "GPT-2": "results/hvb/gpt2",
        "BiLSTM": "results/hvb/bilstm",
    }},
}
CONDITIONS = ["rm_fillers_only", "clean_heavy"]
B = 2000
N_PERM = 2000
SEED = 42


def macro_f1(y_true, y_pred, n_labels):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n_labels)), zero_division=0)


def run_one(corpus, model, out_dir, cluster_col):
    _, test_df = load_train_test(corpus)
    test_df[cluster_col] = test_df[cluster_col].astype(str)
    cluster_ids = test_df[cluster_col].to_numpy()

    y_test = np.load(os.path.join(out_dir, "y_test_ids.npy"))
    if len(y_test) != len(cluster_ids):
        print(f"  SKIP {corpus}/{model}: length mismatch preds={len(y_test)} test_df={len(cluster_ids)}")
        return None
    label2id = json.load(open(os.path.join(out_dir, "label2id.json")))
    n_labels = len(label2id)

    pred_gold = np.load(os.path.join(out_dir, "preds_gold_text.npy"))
    gold_correct_check = (pred_gold == y_test).mean()

    rng = np.random.default_rng(SEED)
    unique_c = np.unique(cluster_ids)
    c_index = {c: np.where(cluster_ids == c)[0] for c in unique_c}

    result = {"corpus": corpus, "model": model, "n_test": int(len(y_test)),
              "n_clusters": int(len(unique_c)), "gold_accuracy_sanity": float(gold_correct_check)}

    for cond in CONDITIONS:
        pred_path = os.path.join(out_dir, f"preds_{cond}.npy")
        if not os.path.exists(pred_path):
            continue
        pred_cond = np.load(pred_path)
        observed_delta = macro_f1(y_test, pred_gold, n_labels) - macro_f1(y_test, pred_cond, n_labels)

        null_deltas = np.empty(N_PERM)
        for i in range(N_PERM):
            swapped_gold = pred_gold.copy()
            swapped_cond = pred_cond.copy()
            for c, idx in c_index.items():
                if rng.random() < 0.5:
                    swapped_gold[idx], swapped_cond[idx] = pred_cond[idx], pred_gold[idx]
            null_deltas[i] = macro_f1(y_test, swapped_gold, n_labels) - macro_f1(y_test, swapped_cond, n_labels)
        p_perm = float(np.mean(np.abs(null_deltas) >= abs(observed_delta)))

        boot_deltas = np.empty(B)
        for i in range(B):
            sample_c = rng.choice(unique_c, size=len(unique_c), replace=True)
            idx = np.concatenate([c_index[c] for c in sample_c])
            yt = y_test[idx]
            boot_deltas[i] = macro_f1(yt, pred_gold[idx], n_labels) - macro_f1(yt, pred_cond[idx], n_labels)

        result[cond] = {
            "observed_delta": float(observed_delta),
            "perm_p_two_sided": p_perm,
            "bootstrap_mean": float(boot_deltas.mean()),
            "bootstrap_ci_lo": float(np.percentile(boot_deltas, 2.5)),
            "bootstrap_ci_hi": float(np.percentile(boot_deltas, 97.5)),
        }
        print(f"  {corpus}/{model}/{cond}: delta={observed_delta:.4f} perm_p={p_perm:.4f} "
              f"boot=[{result[cond]['bootstrap_ci_lo']:.4f}, {result[cond]['bootstrap_ci_hi']:.4f}]")

    return result


def main():
    all_results = []
    for corpus, cfg in CORPUS_CFG.items():
        print(f"\n=== {corpus} (cluster col: {cfg['cluster_col']}) ===")
        for model, out_dir in cfg["models"].items():
            r = run_one(corpus, model, out_dir, cfg["cluster_col"])
            if r:
                all_results.append(r)

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {OUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
