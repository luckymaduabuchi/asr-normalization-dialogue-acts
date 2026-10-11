#!/usr/bin/env python3
"""
Run TF-IDF + Logistic Regression on multiple text conditions and SAVE:
- Core metrics (macro-F1, micro-F1, accuracy) to CSV + JSON
- Per-class report (precision/recall/F1/support) to CSV
- Confusion matrix (raw + normalized) plots for each condition
- Summary bar plot comparing conditions

Usage:
  python run_conditions.py

Inputs:
  utterances_text_conditions.csv   (must contain: conversation_id, label, and the text columns)
Outputs (created in ./results/):
  metrics_summary.csv
  metrics_summary.json
  per_class_report_<condition>.csv
  confusion_<condition>.png
  confusion_norm_<condition>.png
  condition_compare_macro_f1.png
  condition_compare_micro_f1.png
  condition_compare_accuracy.png
  model_<condition>.joblib
  split_conversations.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------
# Config
# -----------------------
DATA = "utterances_text_conditions.csv"
OUT_DIR = "results"
RANDOM_STATE = 42

MIN_COUNT = 50          # drop labels with < MIN_COUNT occurrences
TEST_SIZE = 0.20        # conversation-level test split

CONDITIONS = [
    ("gold_text", "gold_text"),
    ("gold_alpha_text", "gold_alpha_text"),
    ("clean_light", "clean_light"),
    ("clean_heavy", "clean_heavy"),
]

TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
)

LR_KWARGS = dict(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced",
)

# -----------------------
# Helpers
# -----------------------
def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def plot_confusion(cm, labels, title, out_path, normalize=False):
    """
    cm: confusion matrix (raw counts)
    normalize: if True, row-normalize (per true class)
    """
    mat = cm.astype(float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def barplot_conditions(df_metrics, metric_col, out_path, title):
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(df_metrics["condition"], df_metrics[metric_col])
    ax.set_title(title)
    ax.set_xlabel("Condition")
    ax.set_ylabel(metric_col)
    ax.set_xticklabels(df_metrics["condition"], rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])

def run_condition(train_df, test_df, col, labels_order):
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

    # per-class report (dict)
    report_dict = classification_report(
        y_test, pred,
        labels=labels_order,
        output_dict=True,
        zero_division=0
    )

    # confusion matrix (raw)
    cm = confusion_matrix(y_test, pred, labels=labels_order)

    return {
        "model": clf,
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "accuracy": float(acc),
        "report_dict": report_dict,
        "confusion": cm,
    }

# -----------------------
# Main
# -----------------------
def main():
    ensure_out_dir(OUT_DIR)

    df = pd.read_csv(DATA)

    # Drop missing labels
    df = df.dropna(subset=["label"]).copy()

    # Drop rare labels for stability
    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    df = df[df["label"].isin(keep)].copy()

    # Conversation-level split
    conv_ids = df["conversation_id"].unique()
    train_convs, test_convs = train_test_split(
        conv_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_df = df[df["conversation_id"].isin(train_convs)].copy()
    test_df  = df[df["conversation_id"].isin(test_convs)].copy()

    # Save split info so you can reproduce
    split_info = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "min_count": MIN_COUNT,
        "n_conversations_total": int(len(conv_ids)),
        "n_conversations_train": int(len(train_convs)),
        "n_conversations_test": int(len(test_convs)),
        "train_conversation_ids": [str(x) for x in train_convs],
        "test_conversation_ids": [str(x) for x in test_convs],
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
    }
    save_json(split_info, os.path.join(OUT_DIR, "split_conversations.json"))

    # Fix a consistent label order (important for confusion matrices)
    labels_order = sorted(train_df["label"].unique().tolist())

    all_metrics = []
    all_json = {
        "dataset": DATA,
        "n_rows_after_filtering": int(len(df)),
        "n_labels": int(len(labels_order)),
        "labels_order": labels_order,
        "conditions": [],
    }

    for cond_name, col in CONDITIONS:
        if col not in df.columns:
            print(f"[WARN] Missing column '{col}' for condition '{cond_name}'. Skipping.")
            continue

        print(f"\nRunning condition: {cond_name} (column='{col}')")

        out = run_condition(train_df, test_df, col, labels_order)

        # Save model
        model_path = os.path.join(OUT_DIR, f"model_{cond_name}.joblib")
        joblib.dump(out["model"], model_path)

        # Save per-class report to CSV
        # classification_report dict has keys for each label + "accuracy"/"macro avg"/"weighted avg"
        report_df = pd.DataFrame(out["report_dict"]).T
        report_path = os.path.join(OUT_DIR, f"per_class_report_{cond_name}.csv")
        report_df.to_csv(report_path, index=True)

        # Confusion matrix plots
        cm = out["confusion"]
        plot_confusion(
            cm,
            labels_order,
            title=f"Confusion Matrix (raw) - {cond_name}",
            out_path=os.path.join(OUT_DIR, f"confusion_{cond_name}.png"),
            normalize=False
        )
        plot_confusion(
            cm,
            labels_order,
            title=f"Confusion Matrix (normalized) - {cond_name}",
            out_path=os.path.join(OUT_DIR, f"confusion_norm_{cond_name}.png"),
            normalize=True
        )

        # Metrics row
        row = {
            "condition": cond_name,
            "column": col,
            "macro_f1": out["macro_f1"],
            "micro_f1": out["micro_f1"],
            "accuracy": out["accuracy"],
            "model_path": model_path,
            "report_path": report_path,
            "confusion_path": os.path.join(OUT_DIR, f"confusion_{cond_name}.png"),
            "confusion_norm_path": os.path.join(OUT_DIR, f"confusion_norm_{cond_name}.png"),
        }
        all_metrics.append(row)

        # JSON entry
        all_json["conditions"].append({
            "condition": cond_name,
            "column": col,
            "macro_f1": out["macro_f1"],
            "micro_f1": out["micro_f1"],
            "accuracy": out["accuracy"],
            "model_path": model_path,
            "per_class_report_csv": report_path,
            "confusion_png": row["confusion_path"],
            "confusion_norm_png": row["confusion_norm_path"],
        })

        print(f"  Macro-F1={out['macro_f1']:.4f}  Micro-F1={out['micro_f1']:.4f}  Acc={out['accuracy']:.4f}")

    # Save metrics summary
    metrics_df = pd.DataFrame(all_metrics).sort_values("macro_f1", ascending=False)
    metrics_csv = os.path.join(OUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    metrics_json = os.path.join(OUT_DIR, "metrics_summary.json")
    save_json(all_json, metrics_json)

    # Comparison plots
    if len(metrics_df) > 0:
        barplot_conditions(
            metrics_df, "macro_f1",
            os.path.join(OUT_DIR, "condition_compare_macro_f1.png"),
            "Macro-F1 by Condition (conversation-level split)"
        )
        barplot_conditions(
            metrics_df, "micro_f1",
            os.path.join(OUT_DIR, "condition_compare_micro_f1.png"),
            "Micro-F1 by Condition (conversation-level split)"
        )
        barplot_conditions(
            metrics_df, "accuracy",
            os.path.join(OUT_DIR, "condition_compare_accuracy.png"),
            "Accuracy by Condition (conversation-level split)"
        )

    # Print summary
    print("\n=== Results saved to ./results/ ===")
    print("Saved:", metrics_csv)
    print("Saved:", metrics_json)
    print("\n=== Summary (sorted by macro_f1) ===")
    for _, r in metrics_df.iterrows():
        print(f"{r['condition']:16s}  Macro-F1={r['macro_f1']:.4f}  Micro-F1={r['micro_f1']:.4f}  Acc={r['accuracy']:.4f}")

if __name__ == "__main__":
    main()
