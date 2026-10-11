#!/usr/bin/env python3
"""
Conversation-level (issue_id) cluster-aware significance testing for HVB,
mirroring significance_tests.py's Switchboard methodology but clustering by
issue_id instead of Switchboard conversation_id. Adds what the paper
currently lacks for HVB: a cluster-level permutation test and bootstrap CI
on the Macro-F1 delta (McNemar's alone only tests paired utterance-level
correctness, not the Macro-F1 statistic itself, and both ignore that
utterances in the same call are not independent draws).

Fits TF-IDF+LR once per condition on the existing train/test split
(results_hvb/utterances_hvb.csv, 'split' column; issue-disjoint by
construction, verified: 0 overlapping issue_ids between train and test).

Outputs -> results_significance_hvb/
"""
import os
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

DATA = "results_hvb/utterances_hvb.csv"
OUT_DIR = "results_significance_hvb"
BASELINE = "gold_text"
CONDITIONS = ["gold_text", "rm_fillers_only", "clean_heavy"]
B = 2000
N_PERM = 2000
SEED = 42

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])


def macro_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA).dropna(subset=["label"]).copy()
    df["issue_id"] = df["issue_id"].astype(str)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    assert len(set(train_df.issue_id) & set(test_df.issue_id)) == 0, "issue_id leakage between splits"

    labels_order = sorted(train_df["label"].unique().tolist())
    y_true = test_df["label"].to_numpy()
    test_issue_ids = test_df["issue_id"].to_numpy()
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}  Test issues: {test_df.issue_id.nunique()}")

    preds = {}
    for c in CONDITIONS:
        X_train, y_train = train_df[c].fillna(""), train_df["label"]
        X_test = test_df[c].fillna("")
        clf = build_model()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        preds[c] = pred
        print(f"{c:16s} Macro-F1={macro_f1(y_true, pred, labels_order):.4f}  Acc={accuracy_score(y_true, pred):.4f}")

    base_pred = preds[BASELINE]
    base_correct = (base_pred == y_true)

    # ---- McNemar's (paired, utterance-level; reproduces existing table numbers) ----
    mcnemar_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        other_correct = (preds[c] == y_true)
        both_correct = int(np.sum(base_correct & other_correct))
        base_only = int(np.sum(base_correct & ~other_correct))
        other_only = int(np.sum(~base_correct & other_correct))
        both_wrong = int(np.sum(~base_correct & ~other_correct))
        table = [[both_correct, base_only], [other_only, both_wrong]]
        use_exact = (base_only + other_only) < 5000
        result = mcnemar(table, exact=use_exact, correction=not use_exact)
        mcnemar_rows.append({
            "comparison": f"{BASELINE} vs {c}", "both_correct": both_correct,
            "baseline_only_correct": base_only, "other_only_correct": other_only,
            "both_wrong": both_wrong, "statistic": float(result.statistic),
            "p_value": float(result.pvalue), "exact": use_exact,
        })
        print(f"McNemar {BASELINE} vs {c}: p={result.pvalue:.6g}")
    pd.DataFrame(mcnemar_rows).to_csv(os.path.join(OUT_DIR, "mcnemar_tests.csv"), index=False)

    # ---- Issue-level permutation test on Macro-F1 delta ----
    rng = np.random.default_rng(SEED)
    unique_issues = np.unique(test_issue_ids)
    issue_index = {c: np.where(test_issue_ids == c)[0] for c in unique_issues}

    perm_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        other_pred = preds[c]
        observed_delta = macro_f1(y_true, base_pred, labels_order) - macro_f1(y_true, other_pred, labels_order)
        null_deltas = np.empty(N_PERM)
        for i in range(N_PERM):
            swapped_base = base_pred.copy()
            swapped_other = other_pred.copy()
            for issue, idx in issue_index.items():
                if rng.random() < 0.5:
                    swapped_base[idx], swapped_other[idx] = other_pred[idx], base_pred[idx]
            null_deltas[i] = macro_f1(y_true, swapped_base, labels_order) - macro_f1(y_true, swapped_other, labels_order)
        p_two_sided = float(np.mean(np.abs(null_deltas) >= abs(observed_delta)))
        perm_rows.append({
            "comparison": f"{BASELINE} - {c}", "observed_macro_f1_delta": float(observed_delta),
            "n_permutations": N_PERM, "p_two_sided": p_two_sided,
        })
        print(f"Permutation test {BASELINE} - {c}: delta={observed_delta:.4f}, p={p_two_sided:.4f}")
    pd.DataFrame(perm_rows).to_csv(os.path.join(OUT_DIR, "permutation_tests.csv"), index=False)

    # ---- Bootstrap CIs by resampling issues (cached predictions, no refit) ----
    issue_to_rows = issue_index
    boot_metric_samples = {c: {"macro_f1": [], "micro_f1": [], "accuracy": []} for c in CONDITIONS}
    for b in range(B):
        sample_issues = rng.choice(unique_issues, size=len(unique_issues), replace=True)
        idx = np.concatenate([issue_to_rows[c] for c in sample_issues])
        yt = y_true[idx]
        for c in CONDITIONS:
            yp = preds[c][idx]
            boot_metric_samples[c]["macro_f1"].append(f1_score(yt, yp, average="macro", labels=labels_order, zero_division=0))
            boot_metric_samples[c]["micro_f1"].append(f1_score(yt, yp, average="micro", labels=labels_order, zero_division=0))
            boot_metric_samples[c]["accuracy"].append(accuracy_score(yt, yp))

    delta_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        row = {"delta": f"{BASELINE} - {c}"}
        for metric in ["macro_f1", "micro_f1", "accuracy"]:
            base_arr = np.array(boot_metric_samples[BASELINE][metric])
            other_arr = np.array(boot_metric_samples[c][metric])
            diff = base_arr - other_arr
            row[f"mean_{metric}"] = float(diff.mean())
            row[f"lo_{metric}"] = float(np.percentile(diff, 2.5))
            row[f"hi_{metric}"] = float(np.percentile(diff, 97.5))
        delta_rows.append(row)
        print(f"Bootstrap delta {BASELINE} - {c}: macro_f1={row['mean_macro_f1']:.4f} "
              f"[{row['lo_macro_f1']:.4f}, {row['hi_macro_f1']:.4f}]")
    pd.DataFrame(delta_rows).to_csv(os.path.join(OUT_DIR, "bootstrap_deltas.csv"), index=False)

    print("\nSaved all outputs to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
