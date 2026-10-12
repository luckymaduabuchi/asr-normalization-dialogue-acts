#!/usr/bin/env python3
"""
Adds a pragmatics-preserving normalization condition (`pragmatic_norm`) to
each corpus: standardizes colloquial spelling (gonna/wanna/kinda) and casing,
same as the colloquial-normalization step already used elsewhere in this
codebase, but explicitly does NOT strip punctuation and does NOT remove
filled pauses or backchannel tokens -- the two operations chiefly responsible
for the paper's reported Macro-F1 declines. Intended as a candidate "safer
default" to contrast against clean_heavy: does light cosmetic normalization
recover most of clean_heavy's lost performance while still doing some
cleanup?
"""
import re
import pandas as pd


def pragmatic_norm(s: str) -> str:
    s = s or ""
    s = re.sub(r"\bgonna\b", "going to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwanna\b", "want to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bkinda\b", "kind of", s, flags=re.IGNORECASE)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


TARGETS = [
    "utterances_with_ablations.csv",
    "results/mrda/tfidf/utterances_mrda.csv",
    "results/hvb/tfidf/utterances_hvb.csv",
]

for path in TARGETS:
    df = pd.read_csv(path)
    df["pragmatic_norm"] = df["gold_text"].astype(str).map(pragmatic_norm)
    df.to_csv(path, index=False)
    n_changed = (df["pragmatic_norm"] != df["gold_text"].astype(str)).mean()
    print(f"{path}: {df.shape}, pragmatic_norm differs from gold_text in {n_changed:.1%} of rows")
