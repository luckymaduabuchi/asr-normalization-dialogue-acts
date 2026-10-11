#!/usr/bin/env python3
"""
Meeting-clustered permutation testing + bootstrap CIs for the five neural
models on the MRDA meeting-disjoint split, completing the cluster-level
testing extension started by significance_tests_neural_clustered.py (which
covered Switchboard/HVB only). gold_text predictions come from
results_mrda_meetingsplit/preds_<model>_gold.npy (the meeting-disjoint
real-ASR experiment); rm_fillers_only/clean_heavy predictions come from
results_mrda_meetingsplit_conditions/<model>/ (trained separately since the
real-ASR experiment only covered gold_text vs. asr_real). Both sets of
y_test_ids.npy were verified byte-identical in row order before writing this
script, and the "meeting" column is re-derived in that same row order via
train_transformer_worker.load_train_test("mrda_meetingdisjoint").

Output: results_significance_neural/mrda_all_results.json
"""
import os
import json
import numpy as np
from sklearn.metrics import f1_score

from train_transformer_worker import load_train_test

OUT_DIR = "results_significance_neural"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "RoBERTa": "roberta-base",
    "BERT": "bert-base-cased",
    "ALBERT": "albert-base-v2",
    "GPT-2": "gpt2",
    "BiLSTM": "bilstm",
}
CONDITIONS = ["rm_fillers_only", "clean_heavy"]
B = 2000
N_PERM = 2000
SEED = 42

GOLD_DIR = "results_mrda_meetingsplit"
COND_DIR = "results_mrda_meetingsplit_conditions"


def macro_f1(y_true, y_pred, n_labels):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n_labels)), zero_division=0)


def run_one(display_name, model_key):
    _, test_df = load_train_test("mrda_meetingdisjoint")
    cluster_ids = test_df["meeting"].astype(str).to_numpy()

    y_test = np.load(os.path.join(GOLD_DIR, "y_test_ids.npy"))
    y_test_cond = np.load(os.path.join(COND_DIR, model_key, "y_test_ids.npy"))
    assert len(y_test) == len(cluster_ids) == len(y_test_cond)
    assert np.array_equal(y_test, y_test_cond), f"{model_key}: gold/cond y_test_ids mismatch"

    label2id = json.load(open(os.path.join(COND_DIR, model_key, "label2id.json")))
    n_labels = len(label2id)

    pred_gold = np.load(os.path.join(GOLD_DIR, f"preds_{model_key}_gold.npy"))

    rng = np.random.default_rng(SEED)
    unique_c = np.unique(cluster_ids)
    c_index = {c: np.where(cluster_ids == c)[0] for c in unique_c}

    result = {"corpus": "mrda", "model": display_name, "n_test": int(len(y_test)),
              "n_clusters": int(len(unique_c))}

    for cond in CONDITIONS:
        pred_cond = np.load(os.path.join(COND_DIR, model_key, f"preds_{cond}.npy"))
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
        print(f"  mrda/{display_name}/{cond}: delta={observed_delta:.4f} perm_p={p_perm:.4f} "
              f"boot=[{result[cond]['bootstrap_ci_lo']:.4f}, {result[cond]['bootstrap_ci_hi']:.4f}]")

    return result


def main():
    print("\n=== mrda (cluster col: meeting) ===")
    all_results = []
    for display_name, model_key in MODELS.items():
        all_results.append(run_one(display_name, model_key))

    with open(os.path.join(OUT_DIR, "mrda_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {OUT_DIR}/mrda_all_results.json")


if __name__ == "__main__":
    main()
