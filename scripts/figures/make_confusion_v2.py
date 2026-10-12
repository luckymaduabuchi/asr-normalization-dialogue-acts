#!/usr/bin/env python3
"""
Generates the confusion-matrix panels as SIX SEPARATE images (reverting to
the individual-image layout), each still using 'Blues' (0=white, 1=dark blue)
instead of viridis so near-zero off-diagonal cells fade into the page rather
than reading as a solid competing color, each with a fixed 0-1 scale (so
panels stay visually comparable even without one shared colorbar), and each
annotated with the actual numeric value per cell -- including Switchboard
(27x27), using a larger canvas and smaller font so it stays legible.
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
OUT_DIR = "results/cross_corpus/significance"


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])


def normalize(cm):
    mat = cm.astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def plot_confusion(cm, labels, title, out_path):
    mat = normalize(cm)
    n = len(labels)
    large = n > 20
    figsize = (13, 11) if large else (7, 6)
    fontsize_ticks = 6.5 if large else 8
    fontsize_annot = 4.3 if large else 6

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize_ticks)
    ax.set_yticklabels(labels, fontsize=fontsize_ticks)

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if v < 0.005:
                continue
            color = "white" if v > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=fontsize_annot, color=color)

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="black", linewidth=0.6))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def switchboard_cms():
    split = json.load(open("results/switchboard/tfidf/split_conversations.json"))
    test_ids = {str(x) for x in split["test_conversation_ids"]}

    df = pd.read_csv("utterances_with_ablations.csv")
    df = df.dropna(subset=["label"]).copy()
    vc = df["label"].value_counts()
    keep = vc[vc >= 50].index
    df = df[df["label"].isin(keep)].copy()

    test_df = df[df["conversation_id"].astype(str).isin(test_ids)].copy()
    labels_order = sorted(df["label"].unique().tolist())
    y_test = test_df["label"].to_numpy()

    out = {}
    for cond in ["gold_text", "rm_fillers_only"]:
        model = joblib.load(f"results/switchboard/tfidf/model_{cond}.joblib")
        pred = model.predict(test_df[cond].fillna(""))
        out[cond] = (confusion_matrix(y_test, pred, labels=labels_order), labels_order)
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
        out[cond] = (confusion_matrix(y_test, pred, labels=labels_order), labels_order)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Switchboard (reusing saved models)...")
    swb = switchboard_cms()
    for cond, (cm, labels) in swb.items():
        plot_confusion(cm, labels, f"Switchboard: {cond}",
                        os.path.join(OUT_DIR, f"confusion_norm_{cond}.png"))

    print("MRDA...")
    mrda = corpus_cms("results/mrda/tfidf/utterances_mrda.csv")
    for cond, (cm, labels) in mrda.items():
        plot_confusion(cm, labels, f"MRDA: {cond}",
                        os.path.join(OUT_DIR, f"confusion_norm_mrda_{cond}.png"))

    print("HVB...")
    hvb = corpus_cms("results/hvb/tfidf/utterances_hvb.csv")
    for cond, (cm, labels) in hvb.items():
        plot_confusion(cm, labels, f"HVB: {cond}",
                        os.path.join(OUT_DIR, f"confusion_norm_hvb_{cond}.png"))

    print("Done.")


if __name__ == "__main__":
    main()
