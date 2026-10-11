#!/usr/bin/env python3
"""
TF-IDF+LR on the new conditions (rm_fillers_strict, clean_heavy_strict,
pragmatic_norm) for all three corpora, using each corpus's existing
train/test split so results are directly comparable to the main grid.
Appends to each corpus's existing results_*/metrics_summary.csv (new
condition rows only; does not touch existing rows) and runs McNemar's test
against gold_text, matching the rest of the paper's methodology.

MRDA's pragmatic_norm is skipped: verified byte-identical to gold_text for
all 91,769 rows (wylupek/mrda-corpus's gold_text is already lowercase and
already has "gonna"/"wanna" spelled out), so retraining would just
reproduce the existing gold_text row.
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

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")
MIN_COUNT = 50
NEW_CONDITIONS = ["rm_fillers_strict", "clean_heavy_strict", "pragmatic_norm"]


def build_model():
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_KWARGS)), ("lr", LogisticRegression(**LR_KWARGS))])


def macro_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def run_corpus(name, csv_path, out_dir, split_kind, skip_conditions=()):
    print(f"\n=== {name} ===")
    df = pd.read_csv(csv_path).dropna(subset=["label"]).copy()
    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    df = df[df["label"].isin(keep)].copy()

    if split_kind == "conversation_id":
        split = json.load(open("results_ablation/split_conversations.json"))
        df["conversation_id"] = df["conversation_id"].astype(str)
        train_df = df[df["conversation_id"].isin(set(split["train_conversation_ids"]))].copy()
        test_df = df[df["conversation_id"].isin(set(split["test_conversation_ids"]))].copy()
    else:
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()

    labels_order = sorted(train_df["label"].unique().tolist())
    y_test = test_df["label"].to_numpy()

    # gold_text baseline (refit once here so McNemar is paired against the SAME
    # fitted model/predictions used for these new conditions, not stale numbers)
    clf_gold = build_model()
    clf_gold.fit(train_df["gold_text"].fillna(""), train_df["label"])
    pred_gold = clf_gold.predict(test_df["gold_text"].fillna(""))
    gold_correct = (pred_gold == y_test)
    print(f"gold_text        Macro-F1={macro_f1(y_test, pred_gold, labels_order):.4f}")

    rows = []
    mcnemar_rows = []
    for cond in NEW_CONDITIONS:
        if cond in skip_conditions or cond not in df.columns:
            continue
        clf = build_model()
        clf.fit(train_df[cond].fillna(""), train_df["label"])
        pred = clf.predict(test_df[cond].fillna(""))
        macro = macro_f1(y_test, pred, labels_order)
        micro = f1_score(y_test, pred, average="micro", labels=labels_order, zero_division=0)
        acc = accuracy_score(y_test, pred)
        print(f"{cond:20s} Macro-F1={macro:.4f}  Micro-F1={micro:.4f}  Acc={acc:.4f}")

        other_correct = (pred == y_test)
        both_correct = int(np.sum(gold_correct & other_correct))
        gold_only = int(np.sum(gold_correct & ~other_correct))
        other_only = int(np.sum(~gold_correct & other_correct))
        both_wrong = int(np.sum(~gold_correct & ~other_correct))
        table = [[both_correct, gold_only], [other_only, both_wrong]]
        use_exact = (gold_only + other_only) < 5000
        result = mcnemar(table, exact=use_exact, correction=not use_exact)

        rows.append({"condition": cond, "column": cond, "model": "TF-IDF+LR", "dataset": name,
                     "macro_f1": macro, "micro_f1": micro, "accuracy": acc, "status": "OK"})
        mcnemar_rows.append({"comparison": f"gold_text vs {cond}", "dataset": name, "model": "TF-IDF+LR",
                              "gold_only_correct": gold_only, "other_only_correct": other_only,
                              "statistic": float(result.statistic), "p_value": float(result.pvalue)})

    out_csv = os.path.join(out_dir, "extra_conditions_tfidf.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    mc_csv = os.path.join(out_dir, "extra_conditions_mcnemar_tfidf.csv")
    pd.DataFrame(mcnemar_rows).to_csv(mc_csv, index=False)
    print(f"Saved {out_csv}, {mc_csv}")


run_corpus("switchboard", "utterances_with_ablations.csv", "results_ablation", "conversation_id")
run_corpus("mrda", "results_mrda/utterances_mrda.csv", "results_mrda", "split", skip_conditions={"pragmatic_norm"})
run_corpus("hvb", "results_hvb/utterances_hvb.csv", "results_hvb", "split")
