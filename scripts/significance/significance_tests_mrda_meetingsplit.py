#!/usr/bin/env python3
"""
Meeting-level cluster-aware significance testing for MRDA, using the new
self-built meeting-disjoint split (results/mrda/meetingsplit/split_meetings.json)
instead of wylupek/mrda-corpus (which has no meeting field, so this test was
not previously possible for MRDA). Builds rm_fillers_only/clean_heavy on this
dataset using the SAME (non-strict) lexicon as the main grid, fits TF-IDF+LR
once per condition on the 60 train meetings, and runs a meeting-level
permutation test + bootstrap CI on the 15 test meetings, mirroring
scripts/significance/significance_tests.py (Switchboard) and scripts/significance/significance_tests_hvb.py (HVB).

Outputs -> results/cross_corpus/significance_mrda/
"""
import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

SCOPED_CSV = "results/legacy/mrda_real/scoped_utterances.csv"
SPLIT_JSON = "results/mrda/meetingsplit/split_meetings.json"
OUT_DIR = "results/cross_corpus/significance_mrda"
MIN_COUNT = 50
BASELINE = "gold_text"
CONDITIONS = ["gold_text", "rm_fillers_only", "clean_heavy"]
B = 2000
N_PERM = 2000
SEED = 42

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")

FILLERS = set(["um", "uh", "erm", "mm", "hmm", "mhm", "uhhuh", "uh-huh", "huh"])


def light_clean(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rm_fillers_only(s):
    toks = (s or "").split()
    return " ".join(t for t in toks if t.lower().strip(".,?!") not in FILLERS)


def clean_heavy(s):
    s = light_clean(s)
    toks = [t for t in s.split() if t not in FILLERS]
    s = " ".join(toks)
    s = s.replace("gonna", "going to").replace("wanna", "want to").replace("kinda", "kind of")
    return re.sub(r"\s+", " ", s).strip()


def build_model():
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_KWARGS)), ("lr", LogisticRegression(**LR_KWARGS))])


def macro_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(SCOPED_CSV).rename(columns={"general_da": "label"})
    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    df = df[df["label"].isin(keep)].copy()

    df["rm_fillers_only"] = df["gold_text"].astype(str).map(rm_fillers_only)
    df["clean_heavy"] = df["gold_text"].astype(str).map(clean_heavy)

    split = json.load(open(SPLIT_JSON))
    train_meetings = set(split["train_meetings"])
    test_meetings = set(split["test_meetings"])
    train_df = df[df["meeting"].isin(train_meetings)].copy()
    test_df = df[df["meeting"].isin(test_meetings)].copy()
    assert set(train_df["meeting"]).isdisjoint(set(test_df["meeting"]))

    labels_order = sorted(train_df["label"].unique().tolist())
    y_true = test_df["label"].to_numpy()
    test_meeting_ids = test_df["meeting"].to_numpy()
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}  Test meetings: {test_df.meeting.nunique()}")

    preds = {}
    for c in CONDITIONS:
        clf = build_model()
        clf.fit(train_df[c].fillna(""), train_df["label"])
        pred = clf.predict(test_df[c].fillna(""))
        preds[c] = pred
        print(f"{c:16s} Macro-F1={macro_f1(y_true, pred, labels_order):.4f}  Acc={accuracy_score(y_true, pred):.4f}")

    base_pred = preds[BASELINE]
    base_correct = (base_pred == y_true)

    mcnemar_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        other_correct = (preds[c] == y_true)
        both_correct = int(np.sum(base_correct & other_correct))
        base_only = int(np.sum(base_correct & ~other_correct))
        other_only = int(np.sum(~base_correct & other_correct))
        both_wrong = int(np.sum(~base_correct & ~other_correct))
        table = [[both_correct, base_only], [other_only, both_wrong]]
        use_exact = (base_only + other_only) < 5000
        result = mcnemar(table, exact=use_exact, correction=not use_exact)
        mcnemar_rows.append({"comparison": f"{BASELINE} vs {c}", "p_value": float(result.pvalue)})
        print(f"McNemar {BASELINE} vs {c}: p={result.pvalue:.6g}")
    pd.DataFrame(mcnemar_rows).to_csv(os.path.join(OUT_DIR, "mcnemar_tests.csv"), index=False)

    rng = np.random.default_rng(SEED)
    unique_meetings = np.unique(test_meeting_ids)
    meeting_index = {m: np.where(test_meeting_ids == m)[0] for m in unique_meetings}

    perm_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        other_pred = preds[c]
        observed_delta = macro_f1(y_true, base_pred, labels_order) - macro_f1(y_true, other_pred, labels_order)
        null_deltas = np.empty(N_PERM)
        for i in range(N_PERM):
            swapped_base = base_pred.copy()
            swapped_other = other_pred.copy()
            for m, idx in meeting_index.items():
                if rng.random() < 0.5:
                    swapped_base[idx], swapped_other[idx] = other_pred[idx], base_pred[idx]
            null_deltas[i] = macro_f1(y_true, swapped_base, labels_order) - macro_f1(y_true, swapped_other, labels_order)
        p_two_sided = float(np.mean(np.abs(null_deltas) >= abs(observed_delta)))
        perm_rows.append({"comparison": f"{BASELINE} - {c}", "observed_macro_f1_delta": float(observed_delta),
                           "n_permutations": N_PERM, "p_two_sided": p_two_sided})
        print(f"Permutation test {BASELINE} - {c}: delta={observed_delta:.4f}, p={p_two_sided:.4f}")
    pd.DataFrame(perm_rows).to_csv(os.path.join(OUT_DIR, "permutation_tests.csv"), index=False)

    boot_metric_samples = {c: [] for c in CONDITIONS}
    for b in range(B):
        sample_meetings = rng.choice(unique_meetings, size=len(unique_meetings), replace=True)
        idx = np.concatenate([meeting_index[m] for m in sample_meetings])
        yt = y_true[idx]
        for c in CONDITIONS:
            boot_metric_samples[c].append(macro_f1(yt, preds[c][idx], labels_order))

    delta_rows = []
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        base_arr = np.array(boot_metric_samples[BASELINE])
        other_arr = np.array(boot_metric_samples[c])
        diff = base_arr - other_arr
        row = {"delta": f"{BASELINE} - {c}", "mean_macro_f1": float(diff.mean()),
               "lo_macro_f1": float(np.percentile(diff, 2.5)), "hi_macro_f1": float(np.percentile(diff, 97.5))}
        delta_rows.append(row)
        print(f"Bootstrap delta {BASELINE} - {c}: macro_f1={row['mean_macro_f1']:.4f} "
              f"[{row['lo_macro_f1']:.4f}, {row['hi_macro_f1']:.4f}]")
    pd.DataFrame(delta_rows).to_csv(os.path.join(OUT_DIR, "bootstrap_deltas.csv"), index=False)

    print("\nSaved all outputs to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
