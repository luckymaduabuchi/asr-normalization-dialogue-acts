#!/usr/bin/env python3
"""
SUPERSEDED -- do not use for the paper. Kept only for historical reference.

Replaced by significance_tests.py, which fixes two problems with this script:
  1. The "paired sign-style significance test" below is computed as
     (num_nonpos+1)/(n+1) over bootstrap replicates that share the same fitted
     model and largely-overlapping resampled test conversations, so it is not a
     valid independent-samples hypothesis test -- whenever num_nonpos=0 for every
     metric (as happened here), it produces the identical p-value for Macro-F1,
     Micro-F1, and Accuracy, which is a symptom of the method being degenerate,
     not a real per-metric result. significance_tests.py replaces it with
     McNemar's test (paired, per-utterance) and a conversation-level permutation
     test, both computed directly on the fixed test split.
  2. This script's conversation split (np.random.default_rng(42).shuffle) is
     DIFFERENT from the split used everywhere else in the paper (train_ablations.py's
     sklearn train_test_split(..., random_state=42)), so results_bootstrap/*.csv
     report different gold_text point estimates (~0.289 Macro-F1) than the
     ablation table and every other analysis (~0.311 Macro-F1). This was an
     internal inconsistency in the original paper draft. significance_tests.py
     reuses results_ablation/split_conversations.json so every table in the
     paper is computed on the same split.

results_bootstrap/*.csv below are stale for the same reasons and are no longer
cited anywhere in the paper -- see results_significance/ instead.

Original docstring follows:

Bootstrap evaluation (conversation-level resampling) for selected conditions.

Updates vs your version:
- Computes and SAVES relevant metrics: Macro-F1, Micro-F1, Accuracy
- Computes 95% bootstrap CIs for each metric per condition
- Computes deltas vs baseline (gold_text) with CIs
- Computes paired sign-style significance tests over bootstrap replicates
- Saves results to ./results_bootstrap/ :
    - bootstrap_summary.csv
    - bootstrap_deltas.csv
    - significance_tests.csv
    - bootstrap_scores_long.csv  (all bootstrap samples, long format)
    - bootstrap_summary.json
    - optional plots: CI bars per metric (png)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# -----------------------
# Config
# -----------------------
DATA = "utterances_with_ablations.csv"
OUT_DIR = "results_bootstrap"

MIN_COUNT = 50
SEED = 42

# Conditions to bootstrap
CONDITIONS = ["gold_text", "rm_fillers_only", "clean_heavy"]
BASELINE = "gold_text"

# Number of bootstrap samples
B = 300

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")

# -----------------------
# Helpers
# -----------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def ci(arr):
    arr = np.asarray(arr, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])

def fit_and_eval(train_df, test_df, col):
    X_train = train_df[col].fillna("")
    y_train = train_df["label"]
    X_test  = test_df[col].fillna("")
    y_test  = test_df["label"]

    clf = build_model()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    macro = f1_score(y_test, pred, average="macro")
    micro = f1_score(y_test, pred, average="micro")
    acc = accuracy_score(y_test, pred)

    return float(macro), float(micro), float(acc)

def plot_ci_bars(summary_df, metric, out_path, title):
    # Expect columns: condition, mean_<metric>, lo_<metric>, hi_<metric>
    means = summary_df[f"mean_{metric}"].values
    lo = summary_df[f"lo_{metric}"].values
    hi = summary_df[f"hi_{metric}"].values
    x = np.arange(len(summary_df))

    yerr = np.vstack([means - lo, hi - means])

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(x, means)
    ax.errorbar(x, means, yerr=yerr, fmt="none", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["condition"], rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(OUT_DIR)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(DATA).dropna(subset=["label"]).copy()

    # Remove rare labels
    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    df = df[df["label"].isin(keep)].copy()

    # Fixed train/test split by conversation (same style you used)
    conv_ids = df["conversation_id"].unique()
    rng.shuffle(conv_ids)
    split = int(0.8 * len(conv_ids))
    train_convs = set(conv_ids[:split])
    test_convs  = np.array(conv_ids[split:])

    train_df = df[df["conversation_id"].isin(train_convs)].copy()
    test_df  = df[df["conversation_id"].isin(test_convs)].copy()

    # Sanity: ensure all condition columns exist
    missing = [c for c in CONDITIONS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Bootstrap over test conversations
    test_conv_list = test_df["conversation_id"].unique()

    # Store bootstrap samples in long format for easy analysis
    rows_long = []

    for b in range(B):
        sample_convs = rng.choice(test_conv_list, size=len(test_conv_list), replace=True)
        boot_test = test_df[test_df["conversation_id"].isin(sample_convs)].copy()

        for c in CONDITIONS:
            macro, micro, acc = fit_and_eval(train_df, boot_test, c)
            rows_long.append({
                "bootstrap_id": b,
                "condition": c,
                "macro_f1": macro,
                "micro_f1": micro,
                "accuracy": acc,
            })

    long_df = pd.DataFrame(rows_long)
    long_path = os.path.join(OUT_DIR, "bootstrap_scores_long.csv")
    long_df.to_csv(long_path, index=False)

    # Summaries with CIs
    summary_rows = []
    for c in CONDITIONS:
        sub = long_df[long_df["condition"] == c]
        m_macro, lo_macro, hi_macro = ci(sub["macro_f1"])
        m_micro, lo_micro, hi_micro = ci(sub["micro_f1"])
        m_acc, lo_acc, hi_acc = ci(sub["accuracy"])

        summary_rows.append({
            "condition": c,
            "mean_macro_f1": m_macro, "lo_macro_f1": lo_macro, "hi_macro_f1": hi_macro,
            "mean_micro_f1": m_micro, "lo_micro_f1": lo_micro, "hi_micro_f1": hi_micro,
            "mean_accuracy": m_acc, "lo_accuracy": lo_acc, "hi_accuracy": hi_acc,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)
    summary_csv = os.path.join(OUT_DIR, "bootstrap_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Deltas vs baseline (gold_text)
    base = long_df[long_df["condition"] == BASELINE].set_index("bootstrap_id")
    deltas_rows = []

    for c in CONDITIONS:
        if c == BASELINE:
            continue
        other = long_df[long_df["condition"] == c].set_index("bootstrap_id")

        d_macro = base["macro_f1"] - other["macro_f1"]
        d_micro = base["micro_f1"] - other["micro_f1"]
        d_acc = base["accuracy"] - other["accuracy"]

        m, lo, hi = ci(d_macro)
        mm, llo, hhi = ci(d_micro)
        ma, la, ha = ci(d_acc)

        deltas_rows.append({
            "delta": f"{BASELINE} - {c}",
            "mean_macro_f1": m, "lo_macro_f1": lo, "hi_macro_f1": hi,
            "mean_micro_f1": mm, "lo_micro_f1": llo, "hi_micro_f1": hhi,
            "mean_accuracy": ma, "lo_accuracy": la, "hi_accuracy": ha,
        })

    deltas_df = pd.DataFrame(deltas_rows)
    deltas_csv = os.path.join(OUT_DIR, "bootstrap_deltas.csv")
    deltas_df.to_csv(deltas_csv, index=False)

    # Paired sign-style test over bootstrap replicates.
    # We use the paired bootstrap differences already computed above and report
    # a plus-one corrected p-value so that an all-positive result is not shown
    # as zero.
    sig_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        other = long_df[long_df["condition"] == c].set_index("bootstrap_id")
        for metric in ["macro_f1", "micro_f1", "accuracy"]:
            diff = base[metric] - other[metric]
            n = len(diff)
            num_nonpos = int((diff <= 0).sum())
            p_one_sided = (num_nonpos + 1) / (n + 1)
            p_two_sided = min(1.0, 2 * p_one_sided)
            sig_rows.append({
                "condition": c,
                "metric": metric,
                "mean_delta": float(diff.mean()),
                "ci_lo": float(np.percentile(diff, 2.5)),
                "ci_hi": float(np.percentile(diff, 97.5)),
                "num_nonpos": num_nonpos,
                "n_bootstrap": n,
                "p_one_sided": float(p_one_sided),
                "p_two_sided": float(p_two_sided),
            })

    sig_df = pd.DataFrame(sig_rows)
    sig_csv = os.path.join(OUT_DIR, "significance_tests.csv")
    sig_df.to_csv(sig_csv, index=False)

    # JSON bundle
    bundle = {
        "dataset": DATA,
        "seed": SEED,
        "B": B,
        "min_count": MIN_COUNT,
        "conditions": CONDITIONS,
        "baseline": BASELINE,
        "n_rows_after_filtering": int(len(df)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_conversations": int(len(train_convs)),
        "n_test_conversations": int(len(test_convs)),
        "files": {
            "bootstrap_scores_long_csv": long_path,
            "bootstrap_summary_csv": summary_csv,
            "bootstrap_deltas_csv": deltas_csv,
            "significance_tests_csv": sig_csv,
        }
    }
    save_json(bundle, os.path.join(OUT_DIR, "bootstrap_summary.json"))

    # Plots (optional but useful)
    plot_ci_bars(
        summary_df, "macro_f1",
        os.path.join(OUT_DIR, "ci_macro_f1.png"),
        "Bootstrap 95% CI: Macro-F1 by Condition"
    )
    plot_ci_bars(
        summary_df, "micro_f1",
        os.path.join(OUT_DIR, "ci_micro_f1.png"),
        "Bootstrap 95% CI: Micro-F1 by Condition"
    )
    plot_ci_bars(
        summary_df, "accuracy",
        os.path.join(OUT_DIR, "ci_accuracy.png"),
        "Bootstrap 95% CI: Accuracy by Condition"
    )

    # Print console summary
    print("\n=== Bootstrap (mean, 95% CI) ===")
    for _, r in summary_df.iterrows():
        print(
            f"{r['condition']:14s} "
            f"Macro-F1={r['mean_macro_f1']:.4f} [{r['lo_macro_f1']:.4f},{r['hi_macro_f1']:.4f}]   "
            f"Micro-F1={r['mean_micro_f1']:.4f} [{r['lo_micro_f1']:.4f},{r['hi_micro_f1']:.4f}]   "
            f"Acc={r['mean_accuracy']:.4f} [{r['lo_accuracy']:.4f},{r['hi_accuracy']:.4f}]"
        )

    if len(deltas_df) > 0:
        print("\n=== Deltas vs baseline (baseline - condition) ===")
        for _, r in deltas_df.iterrows():
            print(
                f"{r['delta']:28s} "
                f"Macro={r['mean_macro_f1']:.4f} [{r['lo_macro_f1']:.4f},{r['hi_macro_f1']:.4f}]   "
                f"Micro={r['mean_micro_f1']:.4f} [{r['lo_micro_f1']:.4f},{r['hi_micro_f1']:.4f}]   "
                f"Acc={r['mean_accuracy']:.4f} [{r['lo_accuracy']:.4f},{r['hi_accuracy']:.4f}]"
            )

    print("\nSaved all outputs to:", os.path.abspath(OUT_DIR))
    print("Summary CSV:", summary_csv)
    print("Long scores CSV:", long_path)
    print("Deltas CSV:", deltas_csv)

if __name__ == "__main__":
    main()
