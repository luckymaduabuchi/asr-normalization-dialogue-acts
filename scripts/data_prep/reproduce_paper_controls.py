#!/usr/bin/env python3
"""Reproduce the lightweight controls and corpus audits reported in the paper."""

import json
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


OUT_DIR = "results/cross_corpus/controls"
TFIDF = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
FILLERS = {"um", "uh", "erm", "mm", "hmm", "mhm", "uhhuh", "uh-huh", "huh"}


def model(punctuation_aware=False):
    kwargs = dict(TFIDF)
    if punctuation_aware:
        # Corpus punctuation is already whitespace-delimited; preserving those
        # tokens also preserves case and contractions exactly as transcribed.
        kwargs.update(lowercase=False, tokenizer=str.split, token_pattern=None)
    return Pipeline([
        ("tfidf", TfidfVectorizer(**kwargs)),
        ("lr", LogisticRegression(max_iter=2000, n_jobs=1, class_weight="balanced")),
    ])


def load_corpus(name):
    if name == "switchboard":
        df = pd.read_csv("utterances_with_ablations.csv").dropna(subset=["label"])
        counts = df.label.value_counts()
        df = df[df.label.isin(counts[counts >= 50].index)].copy()
        split = json.load(open("results/switchboard/tfidf/split_conversations.json"))
        df["conversation_id"] = df["conversation_id"].astype(str)
        train = df[df.conversation_id.isin(set(split["train_conversation_ids"]))]
        test = df[df.conversation_id.isin(set(split["test_conversation_ids"]))]
    else:
        df = pd.read_csv(f"results_{name}/utterances_{name}.csv").dropna(subset=["label"])
        train, test = df[df.split == "train"], df[df.split == "test"]
    return df, train, test


def score(train, test, column, punctuation_aware=False):
    clf = model(punctuation_aware)
    clf.fit(train[column].fillna(""), train.label)
    pred = clf.predict(test[column].fillna(""))
    labels = sorted(train.label.unique())
    return f1_score(test.label, pred, labels=labels, average="macro", zero_division=0)


def strip_fillers(text):
    return " ".join(t for t in str(text).split() if t.lower().strip(".,?!") not in FILLERS)


def switchboard_heavy_without_extra_discourse(text):
    text = re.sub(r"[^a-z0-9\s']", " ", str(text).lower())
    text = " ".join(t for t in text.split() if t not in FILLERS)
    text = text.replace("gonna", "going to").replace("wanna", "want to").replace("kinda", "kind of")
    return re.sub(r"\s+", " ", text).strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    punctuation_rows = []
    audit_rows = []

    for corpus in ("switchboard", "mrda", "hvb"):
        df, train, test = load_corpus(corpus)
        default_gold = score(train, test, "gold_text")
        default_no_punct = score(train, test, "rm_punct_only")
        aware_gold = score(train, test, "gold_text", True)
        aware_no_punct = score(train, test, "rm_punct_only", True)
        punctuation_rows.append({
            "corpus": corpus,
            "default_gold_macro_f1": default_gold,
            "default_no_punct_macro_f1": default_no_punct,
            "default_delta": default_gold - default_no_punct,
            "aware_gold_macro_f1": aware_gold,
            "aware_no_punct_macro_f1": aware_no_punct,
            "aware_delta": aware_gold - aware_no_punct,
        })

        canonical = df.gold_text.fillna("").str.lower().str.contains(r"\b(?:um|uh|erm)\b", regex=True)
        removed = df.gold_text.fillna("").map(strip_fillers)
        bc = df.label.isin(["b", "backchannel"])
        strict_empty = df.rm_fillers_strict.fillna("").astype(str).str.strip().eq("")
        test_mask = df.split.eq("test") if "split" in df else pd.Series(False, index=df.index)
        audit_rows.append({
            "corpus": corpus,
            "n_utterances": len(df),
            "canonical_filler_utterances": int(canonical.sum()),
            "canonical_filler_rate": canonical.mean(),
            "backchannel_utterances": int(bc.sum()),
            "backchannels_emptied_original_lexicon": int((bc & removed.str.strip().eq("")).sum()),
            "backchannels_emptied_restricted_lexicon": int((bc & strict_empty).sum()),
            "test_backchannel_utterances": int((bc & test_mask).sum()),
            "test_backchannels_emptied_original_lexicon": int((bc & test_mask & removed.str.strip().eq("")).sum()),
            "test_backchannels_emptied_restricted_lexicon": int((bc & test_mask & strict_empty).sum()),
        })

    pd.DataFrame(punctuation_rows).to_csv(f"{OUT_DIR}/punctuation_aware_tfidf.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(f"{OUT_DIR}/lexicon_audit.csv", index=False)

    df, train, test = load_corpus("switchboard")
    alt = "clean_heavy_without_extra_discourse"
    df[alt] = df.gold_text.fillna("").map(switchboard_heavy_without_extra_discourse)
    train = df.loc[train.index]
    test = df.loc[test.index]
    changed = (df[alt] != df.clean_heavy.fillna("")).mean()
    rows = [{
        "condition": "clean_heavy",
        "macro_f1": score(train, test, "clean_heavy"),
        "fraction_text_changed_vs_clean_heavy": 0.0,
    }, {
        "condition": alt,
        "macro_f1": score(train, test, alt),
        "fraction_text_changed_vs_clean_heavy": changed,
    }]
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/switchboard_extra_discourse_control.csv", index=False)


if __name__ == "__main__":
    main()
