#!/usr/bin/env python3
"""
Builds per-label F1 delta (gold_text - rm_fillers_only) tables for MRDA and HVB
across all 6 models, and a per-label delta table for Switchboard across all 6
models (27 labels x 6 models), for the paper's expanded appendix.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

MODELS_DIRS = {
    "switchboard": {
        "TF-IDF+LR": None,  # use per_class_report csvs
        "RoBERTa": "results_transformer",
        "BERT": "results_transformer_bert",
        "ALBERT": "results_transformer_albert",
        "GPT-2": "results_transformer_gpt2",
        "BiLSTM": "results_bilstm",
    },
    "mrda": {
        "TF-IDF+LR": None,
        "RoBERTa": "results_mrda_roberta",
        "BERT": "results_mrda_bert",
        "ALBERT": "results_mrda_albert",
        "GPT-2": "results_mrda_gpt2",
        "BiLSTM": "results_mrda_bilstm",
    },
    "hvb": {
        "TF-IDF+LR": None,
        "RoBERTa": "results_hvb_roberta",
        "BERT": "results_hvb_bert",
        "ALBERT": "results_hvb_albert",
        "GPT-2": "results_hvb_gpt2",
        "BiLSTM": "results_hvb_bilstm",
    },
}
TFIDF_DIRS = {"switchboard": "results_ablation", "mrda": "results_mrda", "hvb": "results_hvb"}


def tfidf_f1(dataset, cond):
    df = pd.read_csv(os.path.join(TFIDF_DIRS[dataset], f"per_class_report_{cond}.csv"), index_col=0)
    df = df.drop([r for r in df.index if r in ("accuracy", "macro avg", "weighted avg")], errors="ignore")
    return df["f1-score"]


def model_f1(out_dir, y_test, label2id, cond):
    preds = np.load(os.path.join(out_dir, f"preds_{cond}.npy"))
    labels_order = [l for l, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
    _, _, f1, _ = precision_recall_fscore_support(y_test, preds, labels=list(range(len(labels_order))), zero_division=0)
    return pd.Series(f1, index=labels_order)


def build(dataset):
    label2id = json.load(open(os.path.join(list(MODELS_DIRS[dataset].values())[1], "label2id.json")))
    y_test = np.load(os.path.join(list(MODELS_DIRS[dataset].values())[1], "y_test_ids.npy"))

    gold_f1 = {}
    fill_f1 = {}
    for model, out_dir in MODELS_DIRS[dataset].items():
        if model == "TF-IDF+LR":
            gold_f1[model] = tfidf_f1(dataset, "gold_text")
            fill_f1[model] = tfidf_f1(dataset, "rm_fillers_only")
        else:
            gold_f1[model] = model_f1(out_dir, y_test, label2id, "gold_text")
            fill_f1[model] = model_f1(out_dir, y_test, label2id, "rm_fillers_only")

    labels = sorted(label2id.keys())
    delta = pd.DataFrame({m: gold_f1[m].reindex(labels) - fill_f1[m].reindex(labels) for m in MODELS_DIRS[dataset]})
    delta.index.name = "label"
    return delta


if __name__ == "__main__":
    os.makedirs("results_significance", exist_ok=True)
    for ds in ["switchboard", "mrda", "hvb"]:
        d = build(ds)
        d.to_csv(f"results_significance/perlabel_delta_{ds}.csv")
        print(f"\n=== {ds} ===")
        print(d.round(3).to_string())
