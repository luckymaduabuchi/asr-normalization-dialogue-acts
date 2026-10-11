#!/usr/bin/env python3
"""
Cross-corpus replication runner for SLUE-HVB, mirroring train_mrda.py's
TF-IDF + Logistic Regression pipeline on the gold_text / rm_fillers_only /
clean_heavy conditions prepared by hvb_prepare.py, using the dataset's own
canonical train/test split. Includes McNemar's test, matching the rigor
applied to Switchboard and MRDA.
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from statsmodels.stats.contingency_tables import mcnemar

DATA = "results_hvb/utterances_hvb.csv"
OUT_DIR = "results_hvb"
CONDITIONS = [
    "gold_text", "gold_alpha_text", "rm_punct_only", "norm_colloq_only",
    "rm_discourse_only", "rm_fillers_only", "clean_light", "clean_heavy", "asr_noise",
]

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])


def main():
    df = pd.read_csv(DATA)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    labels_order = sorted(train_df["label"].unique().tolist())

    rows = []
    preds = {}
    y_test = test_df["label"].to_numpy()
    for c in CONDITIONS:
        X_train = train_df[c].fillna("")
        y_train = train_df["label"]
        X_test = test_df[c].fillna("")

        clf = build_model()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        preds[c] = pred

        macro = f1_score(y_test, pred, average="macro", labels=labels_order, zero_division=0)
        micro = f1_score(y_test, pred, average="micro", labels=labels_order, zero_division=0)
        acc = accuracy_score(y_test, pred)

        report = classification_report(y_test, pred, labels=labels_order, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(os.path.join(OUT_DIR, f"per_class_report_{c}.csv"))

        rows.append({"condition": c, "macro_f1": float(macro), "micro_f1": float(micro), "accuracy": float(acc)})
        print(f"{c:18s} Macro-F1={macro:.4f}  Micro-F1={micro:.4f}  Acc={acc:.4f}")

    base_correct = (preds["gold_text"] == y_test)
    mcnemar_rows = []
    for c in CONDITIONS[1:]:
        other_correct = (preds[c] == y_test)
        base_only = int(np.sum(base_correct & ~other_correct))
        other_only = int(np.sum(~base_correct & other_correct))
        both_correct = int(np.sum(base_correct & other_correct))
        both_wrong = int(np.sum(~base_correct & ~other_correct))
        table = [[both_correct, base_only], [other_only, both_wrong]]
        use_exact = (base_only + other_only) < 5000
        result = mcnemar(table, exact=use_exact, correction=not use_exact)
        mcnemar_rows.append({
            "comparison": f"gold_text vs {c}",
            "baseline_only_correct": base_only,
            "other_only_correct": other_only,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        })
        print(f"McNemar gold_text vs {c}: baseline-only={base_only}, other-only={other_only}, p={result.pvalue:.4g}")
    pd.DataFrame(mcnemar_rows).to_csv(os.path.join(OUT_DIR, "mcnemar_tests.csv"), index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

    baseline = rows[0]
    deltas = []
    for r in rows[1:]:
        deltas.append({
            "delta": f"gold_text - {r['condition']}",
            "macro_f1_delta": baseline["macro_f1"] - r["macro_f1"],
            "micro_f1_delta": baseline["micro_f1"] - r["micro_f1"],
            "accuracy_delta": baseline["accuracy"] - r["accuracy"],
        })
    pd.DataFrame(deltas).to_csv(os.path.join(OUT_DIR, "deltas_summary.csv"), index=False)

    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump({"labels_order": labels_order, "conditions": rows}, f, indent=2)

    print("\nSaved to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
