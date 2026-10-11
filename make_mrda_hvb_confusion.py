#!/usr/bin/env python3
"""
Generates normalized confusion matrices (gold_text vs rm_fillers_only, TF-IDF+LR)
for MRDA and HVB, matching the plotting style already used for Switchboard in
train_ablations.py, so the paper's confusion-matrix figure can show all three
corpora instead of Switchboard only.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")

CORPORA = {
    "mrda": "results_mrda/utterances_mrda.csv",
    "hvb": "results_hvb/utterances_hvb.csv",
}
CONDITIONS = ["gold_text", "rm_fillers_only"]
OUT_DIR = "results_significance"


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(**LR_KWARGS)),
    ])


def plot_confusion(cm, labels, title, out_path):
    mat = cm.astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for corpus, path in CORPORA.items():
        df = pd.read_csv(path)
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
        labels_order = sorted(train_df["label"].unique().tolist())
        y_test = test_df["label"].to_numpy()

        for cond in CONDITIONS:
            X_train = train_df[cond].fillna("")
            y_train = train_df["label"]
            X_test = test_df[cond].fillna("")

            clf = build_model()
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, pred, labels=labels_order)

            out_path = os.path.join(OUT_DIR, f"confusion_norm_{corpus}_{cond}.png")
            plot_confusion(cm, labels_order, f"{corpus.upper()}: {cond}", out_path)
            print(f"[{corpus}] {cond}: saved {out_path}")


if __name__ == "__main__":
    main()
