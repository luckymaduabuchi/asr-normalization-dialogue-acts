#!/usr/bin/env python3
"""
Statistically sound significance testing for gold_text vs. rm_fillers_only / clean_heavy.

Replaces two problems in the original analysis:
  1. bootstrap_ci.py's "sign-style test" over bootstrap replicates is not a real
     per-metric hypothesis test (bootstrap replicates share the same fitted model
     and largely overlapping test conversations, so they are not independent
     draws; the reported p-value only reflects how many of the 300 resamples
     were nonpositive, which is identical across metrics whenever num_nonpos=0).
  2. The paper's z-test on accuracy proportions (z=3.40/3.87) does not exist in
     any script in this repo and, methodologically, an unpaired proportions test
     ignores that predictions for the same utterance across conditions are paired.

This script instead:
  - Fits gold_text / rm_fillers_only / clean_heavy ONCE on the fixed conversation
    split saved by train_ablations.py (results_ablation/split_conversations.json),
    so results are directly comparable to Table tab:ablations.
  - Runs McNemar's test (paired, per-utterance correct/incorrect) for
    gold_text vs rm_fillers_only and gold_text vs clean_heavy.
  - Runs a conversation-level permutation test on the Macro-F1 delta (assumption-light,
    respects the same conversation-level unit used for the train/test split).
  - Recomputes bootstrap CIs by resampling the fixed predictions (no refitting),
    which is both correct and ~300x cheaper than the original approach.
  - Adds a bootstrap CI on the per-label backchannel ('b') F1 delta between
    rm_fillers_only and clean_heavy, to support or soften the paper's "0.093 is a
    substantially larger drop" claim.

Outputs -> results_significance/
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

DATA = "utterances_with_ablations.csv"
SPLIT_FILE = "results_ablation/split_conversations.json"
OUT_DIR = "results_significance"
MIN_COUNT = 50
BASELINE = "gold_text"
CONDITIONS = ["gold_text", "rm_fillers_only", "clean_heavy"]
B = 2000  # bootstrap resamples over test conversations (predictions only, cheap now)
N_PERM = 2000
SEED = 42

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])


def macro_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def main():
    ensure_dir(OUT_DIR)

    df = pd.read_csv(DATA).dropna(subset=["label"]).copy()
    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    df = df[df["label"].isin(keep)].copy()

    split = json.load(open(SPLIT_FILE))
    train_convs = set(split["train_conversation_ids"])
    test_convs = set(split["test_conversation_ids"])
    # conversation_id is read as str already; guard against str/obj mismatch
    df["conversation_id"] = df["conversation_id"].astype(str)

    train_df = df[df["conversation_id"].isin(train_convs)].copy()
    test_df = df[df["conversation_id"].isin(test_convs)].copy()

    labels_order = sorted(train_df["label"].unique().tolist())
    y_true = test_df["label"].to_numpy()
    test_conv_ids = test_df["conversation_id"].to_numpy()

    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}  Test convs: {len(test_convs)}")

    # ---- Fit once per condition, cache predictions ----
    preds = {}
    for c in CONDITIONS:
        X_train = train_df[c].fillna("")
        y_train = train_df["label"]
        X_test = test_df[c].fillna("")

        clf = build_model()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        preds[c] = pred

        print(f"{c:16s} Macro-F1={macro_f1(y_true, pred, labels_order):.4f}  "
              f"Acc={accuracy_score(y_true, pred):.4f}")

    base_pred = preds[BASELINE]
    base_correct = (base_pred == y_true)

    # ---- McNemar's test (paired, per-utterance) ----
    mcnemar_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        other_correct = (preds[c] == y_true)

        # 2x2 table: baseline correct/incorrect x other correct/incorrect
        both_correct = int(np.sum(base_correct & other_correct))
        base_only = int(np.sum(base_correct & ~other_correct))   # baseline right, other wrong
        other_only = int(np.sum(~base_correct & other_correct))  # baseline wrong, other right
        both_wrong = int(np.sum(~base_correct & ~other_correct))

        table = [[both_correct, base_only], [other_only, both_wrong]]
        # exact=True uses the binomial exact test on the discordant pairs, appropriate
        # given base_only/other_only counts here (in the hundreds, not huge)
        use_exact = (base_only + other_only) < 5000
        result = mcnemar(table, exact=use_exact, correction=not use_exact)

        mcnemar_rows.append({
            "comparison": f"{BASELINE} vs {c}",
            "both_correct": both_correct,
            "baseline_only_correct": base_only,
            "other_only_correct": other_only,
            "both_wrong": both_wrong,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "exact": use_exact,
        })
        print(f"McNemar {BASELINE} vs {c}: baseline-only-correct={base_only}, "
              f"{c}-only-correct={other_only}, stat={result.statistic:.4f}, p={result.pvalue:.6g}")

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_df.to_csv(os.path.join(OUT_DIR, "mcnemar_tests.csv"), index=False)

    # ---- Conversation-level permutation test on Macro-F1 delta ----
    rng = np.random.default_rng(SEED)
    unique_convs = np.unique(test_conv_ids)
    conv_index = {c: np.where(test_conv_ids == c)[0] for c in unique_convs}

    perm_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        other_pred = preds[c]
        observed_delta = macro_f1(y_true, base_pred, labels_order) - macro_f1(y_true, other_pred, labels_order)

        null_deltas = np.empty(N_PERM)
        for i in range(N_PERM):
            # For each conversation, independently swap which condition's predictions
            # are labeled "baseline" vs "other" for all utterances in that conversation.
            swapped_base = base_pred.copy()
            swapped_other = other_pred.copy()
            for conv, idx in conv_index.items():
                if rng.random() < 0.5:
                    swapped_base[idx], swapped_other[idx] = other_pred[idx], base_pred[idx]
            null_deltas[i] = macro_f1(y_true, swapped_base, labels_order) - macro_f1(y_true, swapped_other, labels_order)

        p_two_sided = float(np.mean(np.abs(null_deltas) >= abs(observed_delta)))
        perm_rows.append({
            "comparison": f"{BASELINE} - {c}",
            "observed_macro_f1_delta": float(observed_delta),
            "n_permutations": N_PERM,
            "p_two_sided": p_two_sided,
        })
        print(f"Permutation test {BASELINE} - {c}: delta={observed_delta:.4f}, p={p_two_sided:.4f}")

    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(os.path.join(OUT_DIR, "permutation_tests.csv"), index=False)

    # ---- Bootstrap CIs by resampling cached predictions (no refit) ----
    boot_rows = []
    conv_to_rows = {c: np.where(test_conv_ids == c)[0] for c in unique_convs}

    boot_metric_samples = {c: {"macro_f1": [], "micro_f1": [], "accuracy": []} for c in CONDITIONS}
    for b in range(B):
        sample_convs = rng.choice(unique_convs, size=len(unique_convs), replace=True)
        idx = np.concatenate([conv_to_rows[c] for c in sample_convs])
        yt = y_true[idx]
        for c in CONDITIONS:
            yp = preds[c][idx]
            boot_metric_samples[c]["macro_f1"].append(f1_score(yt, yp, average="macro", labels=labels_order, zero_division=0))
            boot_metric_samples[c]["micro_f1"].append(f1_score(yt, yp, average="micro", labels=labels_order, zero_division=0))
            boot_metric_samples[c]["accuracy"].append(accuracy_score(yt, yp))

    summary_rows = []
    for c in CONDITIONS:
        row = {"condition": c}
        for metric in ["macro_f1", "micro_f1", "accuracy"]:
            arr = np.array(boot_metric_samples[c][metric])
            row[f"mean_{metric}"] = float(arr.mean())
            row[f"lo_{metric}"] = float(np.percentile(arr, 2.5))
            row[f"hi_{metric}"] = float(np.percentile(arr, 97.5))
        summary_rows.append(row)
    boot_summary_df = pd.DataFrame(summary_rows)
    boot_summary_df.to_csv(os.path.join(OUT_DIR, "bootstrap_summary.csv"), index=False)

    delta_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        row = {"delta": f"{BASELINE} - {c}"}
        n_favor_baseline = {}
        for metric in ["macro_f1", "micro_f1", "accuracy"]:
            base_arr = np.array(boot_metric_samples[BASELINE][metric])
            other_arr = np.array(boot_metric_samples[c][metric])
            diff = base_arr - other_arr
            row[f"mean_{metric}"] = float(diff.mean())
            row[f"lo_{metric}"] = float(np.percentile(diff, 2.5))
            row[f"hi_{metric}"] = float(np.percentile(diff, 97.5))
            n_favor_baseline[metric] = int(np.sum(diff > 0))
        row["n_bootstrap"] = B
        row["n_favor_baseline_macro_f1"] = n_favor_baseline["macro_f1"]
        row["n_favor_baseline_micro_f1"] = n_favor_baseline["micro_f1"]
        row["n_favor_baseline_accuracy"] = n_favor_baseline["accuracy"]
        delta_rows.append(row)
    boot_delta_df = pd.DataFrame(delta_rows)
    boot_delta_df.to_csv(os.path.join(OUT_DIR, "bootstrap_deltas.csv"), index=False)

    # ---- Per-label backchannel ('b') bootstrap CI: rm_fillers_only vs clean_heavy ----
    target_label = "b"
    if target_label in labels_order:
        b_deltas = np.empty(B)
        rng2 = np.random.default_rng(SEED + 1)
        for i in range(B):
            sample_convs = rng2.choice(unique_convs, size=len(unique_convs), replace=True)
            idx = np.concatenate([conv_to_rows[c] for c in sample_convs])
            yt = y_true[idx]
            p_fill, r_fill, f1_fill, _ = precision_recall_fscore_support(
                yt, preds["rm_fillers_only"][idx], labels=[target_label], zero_division=0)
            p_heavy, r_heavy, f1_heavy, _ = precision_recall_fscore_support(
                yt, preds["clean_heavy"][idx], labels=[target_label], zero_division=0)
            b_deltas[i] = float(f1_fill[0] - f1_heavy[0])

        backchannel_result = {
            "label": target_label,
            "point_estimate_f1_rm_fillers": float(precision_recall_fscore_support(
                y_true, preds["rm_fillers_only"], labels=[target_label], zero_division=0)[2][0]),
            "point_estimate_f1_clean_heavy": float(precision_recall_fscore_support(
                y_true, preds["clean_heavy"], labels=[target_label], zero_division=0)[2][0]),
            "mean_delta_rm_minus_heavy": float(b_deltas.mean()),
            "ci_lo": float(np.percentile(b_deltas, 2.5)),
            "ci_hi": float(np.percentile(b_deltas, 97.5)),
            "n_bootstrap": B,
            "n_favor_rm_fillers": int(np.sum(b_deltas > 0)),
        }
        save_json(backchannel_result, os.path.join(OUT_DIR, "backchannel_delta_ci.json"))
        print(f"\nBackchannel (b) F1 delta (rm_fillers - clean_heavy): "
              f"{backchannel_result['mean_delta_rm_minus_heavy']:.4f} "
              f"[{backchannel_result['ci_lo']:.4f}, {backchannel_result['ci_hi']:.4f}]")

    print("\nSaved all outputs to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
