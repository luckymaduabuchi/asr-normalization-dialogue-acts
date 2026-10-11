#!/usr/bin/env python3
"""
Builds ONE publication-quality confusion-matrix figure (3 corpora x 2 conditions)
with a SHARED color scale (vmin=0, vmax=1) and a single colorbar, replacing the
earlier approach of stitching together six separately-scaled PNGs. Different
color scales across panels in the same figure are misleading for a reader
comparing intensity between panels, which is what this fixes.

Switchboard reuses the already-trained, saved TF-IDF+LR models (no refitting,
so numbers stay identical to what's already reported in the paper). MRDA and
HVB are refit (fast, TF-IDF+LR, CPU, seconds) since no saved model artifacts
exist for them.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])


def switchboard_cms():
    split = json.load(open("results_ablation/split_conversations.json"))
    test_ids = set(split["test_conversation_ids"])

    df = pd.read_csv("utterances_with_ablations.csv")
    df = df.dropna(subset=["label"]).copy()
    vc = df["label"].value_counts()
    keep = vc[vc >= 50].index
    df = df[df["label"].isin(keep)].copy()

    conv_col = "conversation_id" if "conversation_id" in df.columns else "convo_id"
    test_df = df[df[conv_col].astype(str).isin({str(x) for x in test_ids})].copy()
    labels_order = sorted(df["label"].unique().tolist())
    y_test = test_df["label"].to_numpy()

    out = {}
    for cond in ["gold_text", "rm_fillers_only"]:
        model = joblib.load(f"results_ablation/model_{cond}.joblib")
        pred = model.predict(test_df[cond].fillna(""))
        cm = confusion_matrix(y_test, pred, labels=labels_order)
        out[cond] = (cm, labels_order)
    return out


def corpus_cms(path):
    df = pd.read_csv(path)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    labels_order = sorted(train_df["label"].unique().tolist())
    y_test = test_df["label"].to_numpy()

    out = {}
    for cond in ["gold_text", "rm_fillers_only"]:
        clf = build_model()
        clf.fit(train_df[cond].fillna(""), train_df["label"])
        pred = clf.predict(test_df[cond].fillna(""))
        cm = confusion_matrix(y_test, pred, labels=labels_order)
        out[cond] = (cm, labels_order)
    return out


def normalize(cm):
    mat = cm.astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def main():
    print("Computing Switchboard confusion matrices (reusing saved models)...")
    swb = switchboard_cms()
    print("Computing MRDA confusion matrices...")
    mrda = corpus_cms("results_mrda/utterances_mrda.csv")
    print("Computing HVB confusion matrices...")
    hvb = corpus_cms("results_hvb/utterances_hvb.csv")

    rows = [("Switchboard", swb), ("MRDA", mrda), ("HVB", hvb)]
    fig, axes = plt.subplots(3, 2, figsize=(9, 12.5))

    im = None
    for r, (corpus_name, data) in enumerate(rows):
        for c, cond in enumerate(["gold_text", "rm_fillers_only"]):
            cm, labels = data[cond]
            mat = normalize(cm)
            ax = axes[r, c]
            im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            n = len(labels)
            fontsize = 5.5 if n > 20 else (7 if n > 12 else 8)
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
            ax.set_yticklabels(labels, fontsize=fontsize)
            ax.set_title(f"{corpus_name}: {cond}", fontsize=10)
            if r == 2:
                ax.set_xlabel("Predicted label", fontsize=9)
            if c == 0:
                ax.set_ylabel("True label", fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Row-normalized fraction")

    out_path = "results_significance/confusion_unified.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
