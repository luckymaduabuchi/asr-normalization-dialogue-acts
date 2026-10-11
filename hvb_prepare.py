#!/usr/bin/env python3
"""
Second cross-corpus replication: prepare SLUE-HVB (Gridspace-Stanford Harper
Valley Bank call-center dialogues, asapp/slue-phase-2 "hvb" config) with the
same gold_text / rm_fillers_only / clean_heavy conditions used for Switchboard
and MRDA, using the identical filler lexicon and cleaning logic.

Register: synthetic/scripted call-center customer service dialogue (agent vs.
caller), a third and distinct conversational register from Switchboard
(dyadic telephone calls) and MRDA (in-person meetings). Disfluencies are rarer
here (~5.8% of utterances contain um/uh/erm, vs. ~40% in Switchboard) since the
dialogue is more scripted, but real and non-trivial, and the label scheme
includes an explicit "disfluency" dialogue-act category.

Uses the dataset's own canonical train/test split (issue_id-disjoint by
construction, matching the corpus release). dialog_acts is a multi-label
sequence field (~40% of utterances carry >1 label); we take the first label as
the primary label for single-label classification, consistent with how
get_label() in make_text_conditions.py collapses SwDA's multi-tag field.
"""

import os
import re
import json
import pandas as pd
from datasets import load_dataset

OUT_DIR = "results_hvb"
MIN_COUNT = 50

FILLERS = set(["um", "uh", "erm", "mm", "hmm", "mhm", "uhhuh", "uh-huh", "huh"])


def light_clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rm_fillers_only(s: str) -> str:
    toks = (s or "").split()
    toks = [t for t in toks if t.lower().strip(".,?!") not in FILLERS]
    return " ".join(toks)


def heavy_clean(s: str) -> str:
    s = light_clean(s)
    toks = [t for t in s.split() if t not in FILLERS]
    s = " ".join(toks)
    s = s.replace("gonna", "going to").replace("wanna", "want to").replace("kinda", "kind of")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("asapp/slue-phase-2", "hvb")

    def to_df(split_name):
        d = ds[split_name]
        df = pd.DataFrame({
            "issue_id": d["issue_id"],
            "speaker_id": d["speaker_id"],
            "role": d["role"],
            "gold_text": d["text"],
            "label": [labs[0] if len(labs) > 0 else None for labs in d["dialog_acts"]],
        })
        df["split"] = split_name
        return df

    train_df = to_df("train")
    test_df = to_df("test")

    full = pd.concat([train_df, test_df], ignore_index=True)
    full = full.dropna(subset=["label", "gold_text"]).copy()
    full = full[full["gold_text"].str.strip() != ""].copy()

    vc = full["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    full = full[full["label"].isin(keep)].copy()

    full["rm_fillers_only"] = full["gold_text"].map(rm_fillers_only)
    full["clean_heavy"] = full["gold_text"].map(heavy_clean)

    full.to_csv(os.path.join(OUT_DIR, "utterances_hvb.csv"), index=False)

    train_df = full[full["split"] == "train"]
    test_df = full[full["split"] == "test"]

    print(f"Kept labels (min_count={MIN_COUNT}): {sorted(keep.tolist())}")
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")
    print(f"Conversations (issue_id): {full['issue_id'].nunique()} total")

    n_filler = full["gold_text"].str.lower().str.contains(r"\b(um|uh|erm)\b", regex=True).sum()
    print(f"Utterances with um/uh/erm: {n_filler} ({n_filler/len(full):.1%})")

    save_json = {
        "dataset": "asapp/slue-phase-2 (hvb config)",
        "min_count": MIN_COUNT,
        "labels_kept": sorted(keep.tolist()),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "split_note": "Uses the dataset's own canonical train/test split (issue_id-disjoint by construction).",
        "multilabel_note": "dialog_acts is multi-label (~40% of utterances have >1 tag); first label used as primary label.",
    }
    with open(os.path.join(OUT_DIR, "hvb_split_info.json"), "w") as f:
        json.dump(save_json, f, indent=2)


if __name__ == "__main__":
    main()
