#!/usr/bin/env python3
"""
Ablation runner (utterances_with_ablations.csv) with SAVED outputs:
- Core metrics per condition (macro-F1, micro-F1, accuracy) -> results_ablation/metrics_summary.csv + .json
- Per-class precision/recall/F1/support -> results_ablation/per_class_report_<cond>.csv
- Confusion matrix plots (raw + normalized) -> results_ablation/confusion_<cond>.png and confusion_norm_<cond>.png
- Comparison bar plots -> results_ablation/compare_*.png
- Trained model per condition -> results_ablation/model_<cond>.joblib
- Saved split (train/test conversation ids) -> results_ablation/split_conversations.json
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
DATA = "utterances_with_ablations.csv"
OUT_DIR = "results_ablation"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_COUNT = 50

CONDITIONS = [
    ("gold_text", "gold_text"),
    ("gold_alpha_text", "gold_alpha_text"),
    ("rm_punct_only", "rm_punct_only"),
    ("rm_fillers_only", "rm_fillers_only"),
    ("rm_discourse_only", "rm_discourse_only"),
    ("norm_colloq_only", "norm_colloq_only"),
    ("clean_light", "clean_light"),
    ("clean_heavy", "clean_heavy"),
    ("asr_noise", "asr_noise"),
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
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])

def plot_confusion(cm, labels, title, out_path, normalize=False):
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
    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(111)

    ax.bar(df_metrics["condition"], df_metrics[metric_col])
    ax.set_title(title)
    ax.set_xlabel("Condition")
    ax.set_ylabel(metric_col)

    ax.set_xticks(np.arange(len(df_metrics)))
    ax.set_xticklabels(df_metrics["condition"], rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

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

    report_dict = classification_report(
        y_test,
        pred,
        labels=labels_order,
        output_dict=True,
        zero_division=0,
    )

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
    ensure_dir(OUT_DIR)

    df = pd.read_csv(DATA)

    # Drop missing labels
    df = df.dropna(subset=["label"]).copy()

    # Remove rare labels
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

    split_info = {
        "dataset": DATA,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "min_count": MIN_COUNT,
        "n_rows_after_filtering": int(len(df)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_conversations_total": int(len(conv_ids)),
        "n_conversations_train": int(len(train_convs)),
        "n_conversations_test": int(len(test_convs)),
        "train_conversation_ids": [str(x) for x in train_convs],
        "test_conversation_ids": [str(x) for x in test_convs],
    }
    save_json(split_info, os.path.join(OUT_DIR, "split_conversations.json"))

    # Consistent label order
    labels_order = sorted(train_df["label"].unique().tolist())

    metrics_rows = []
    metrics_json = {
        "dataset": DATA,
        "out_dir": OUT_DIR,
        "labels_order": labels_order,
        "split": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "min_count": MIN_COUNT,
        },
        "conditions": [],
    }

    print("\n=== Ablation Results (conversation-level split) ===")

    for cond_name, col in CONDITIONS:
        if col not in df.columns:
            print(f"[WARN] Missing column '{col}' for condition '{cond_name}'. Skipping.")
            continue

        out = run_condition(train_df, test_df, col, labels_order)

        # Save model
        model_path = os.path.join(OUT_DIR, f"model_{cond_name}.joblib")
        joblib.dump(out["model"], model_path)

        # Save per-class report
        report_df = pd.DataFrame(out["report_dict"]).T
        report_path = os.path.join(OUT_DIR, f"per_class_report_{cond_name}.csv")
        report_df.to_csv(report_path, index=True)

        # Confusion matrix plots
        cm = out["confusion"]
        cm_raw_path = os.path.join(OUT_DIR, f"confusion_{cond_name}.png")
        cm_norm_path = os.path.join(OUT_DIR, f"confusion_norm_{cond_name}.png")

        plot_confusion(
            cm,
            labels_order,
            title=f"Confusion Matrix (raw) - {cond_name}",
            out_path=cm_raw_path,
            normalize=False,
        )
        plot_confusion(
            cm,
            labels_order,
            title=f"Confusion Matrix (normalized) - {cond_name}",
            out_path=cm_norm_path,
            normalize=True,
        )

        row = {
            "condition": cond_name,
            "column": col,
            "macro_f1": out["macro_f1"],
            "micro_f1": out["micro_f1"],
            "accuracy": out["accuracy"],
            "model_path": model_path,
            "report_path": report_path,
            "confusion_path": cm_raw_path,
            "confusion_norm_path": cm_norm_path,
        }
        metrics_rows.append(row)

        metrics_json["conditions"].append({
            "condition": cond_name,
            "column": col,
            "macro_f1": out["macro_f1"],
            "micro_f1": out["micro_f1"],
            "accuracy": out["accuracy"],
            "model_path": model_path,
            "per_class_report_csv": report_path,
            "confusion_png": cm_raw_path,
            "confusion_norm_png": cm_norm_path,
        })

        print(f"{cond_name:18s} Macro-F1={out['macro_f1']:.4f}  Micro-F1={out['micro_f1']:.4f}  Acc={out['accuracy']:.4f}")

    # Save metrics summary table
    metrics_df = pd.DataFrame(metrics_rows).sort_values("macro_f1", ascending=False)
    metrics_csv = os.path.join(OUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    save_json(metrics_json, os.path.join(OUT_DIR, "metrics_summary.json"))

    # Comparison plots
    if len(metrics_df) > 0:
        barplot_conditions(
            metrics_df, "macro_f1",
            os.path.join(OUT_DIR, "compare_macro_f1.png"),
            "Macro-F1 by Condition (conversation-level split)"
        )
        barplot_conditions(
            metrics_df, "micro_f1",
            os.path.join(OUT_DIR, "compare_micro_f1.png"),
            "Micro-F1 by Condition (conversation-level split)"
        )
        barplot_conditions(
            metrics_df, "accuracy",
            os.path.join(OUT_DIR, "compare_accuracy.png"),
            "Accuracy by Condition (conversation-level split)"
        )

    print("\nSaved all outputs to:", os.path.abspath(OUT_DIR))
    print("Metrics table:", metrics_csv)

if __name__ == "__main__":
    main()
