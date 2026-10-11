#!/usr/bin/env python3
"""
Adds rm_fillers_strict / clean_heavy_strict columns to each corpus's
existing ablation table, using a STRICT filler lexicon restricted to
canonical hesitation markers (um, uh, erm) and excluding backchannel-
adjacent tokens (mm, hmm, mhm, uh-huh, uhhuh, huh) that the original
FILLERS lexicon also removed. This directly tests whether the paper's
findings survive when filler removal cannot also delete the lexical
realization of the backchannel dialogue act.

Mirrors each corpus's existing light_clean/heavy_clean logic exactly
(same lowercase/punct-strip/colloquial-normalize steps), swapping only
the filler lexicon, so rm_fillers_strict and clean_heavy_strict are
directly comparable to rm_fillers_only and clean_heavy in the main grid.
"""
import re
import pandas as pd

STRICT_FILLERS = set(["um", "uh", "erm"])


def light_clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rm_fillers_strict(s: str) -> str:
    toks = (s or "").split()
    toks = [t for t in toks if t.lower().strip(".,?!") not in STRICT_FILLERS]
    return " ".join(toks)


def clean_heavy_strict(s: str) -> str:
    s = light_clean(s)
    toks = [t for t in s.split() if t not in STRICT_FILLERS]
    s = " ".join(toks)
    s = s.replace("gonna", "going to").replace("wanna", "want to").replace("kinda", "kind of")
    s = re.sub(r"\s+", " ", s).strip()
    return s


TARGETS = [
    ("utterances_with_ablations.csv", "utterances_with_ablations.csv"),
    ("results_mrda/utterances_mrda.csv", "results_mrda/utterances_mrda.csv"),
    ("results_hvb/utterances_hvb.csv", "results_hvb/utterances_hvb.csv"),
]

for src, dst in TARGETS:
    df = pd.read_csv(src)
    df["rm_fillers_strict"] = df["gold_text"].astype(str).map(rm_fillers_strict)
    df["clean_heavy_strict"] = df["gold_text"].astype(str).map(clean_heavy_strict)
    df.to_csv(dst, index=False)
    n_changed_fill = (df["rm_fillers_strict"] != df["rm_fillers_only"]).mean() if "rm_fillers_only" in df.columns else float("nan")
    print(f"{src}: {df.shape}, rm_fillers_strict differs from rm_fillers_only in {n_changed_fill:.1%} of rows")
