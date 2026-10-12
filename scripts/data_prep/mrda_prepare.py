#!/usr/bin/env python3
"""
Cross-corpus replication: prepare the ICSI MRDA corpus (meeting speech) with the
same gold_text / rm_fillers_only / clean_heavy conditions used for Switchboard,
using the identical filler lexicon and cleaning logic from
scripts/data_prep/make_text_conditions.py / scripts/data_prep/make_ablations.py.

Uses wylupek/mrda-corpus on HuggingFace (public, no gating). That release provides
a canonical train/test/validation split that is meeting-disjoint (75 meetings
total; this is the standard split used across the MRDA literature, e.g. Ang,
Liu & Shriberg 2005), so unlike a from-scratch random split it does NOT suffer
from speaker-code reuse across meetings -- checked directly: individual speaker
codes like "fe016" recur across the entire corpus (min/max utterance index span
the full dataset), so meeting id cannot be recovered from the speaker field
alone. We rely on the dataset's own pre-defined split rather than inventing one.

Uses general_da (12-way) as the label column -- comparable granularity to
Switchboard's 27-way scheme after the same min-count filtering.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datasets import load_dataset

OUT_DIR = "results/mrda/tfidf"
MIN_COUNT = 50
LABEL_COL = "general_da"

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
    ds = load_dataset("wylupek/mrda-corpus")

    def to_df(split_name):
        d = ds[split_name]
        df = pd.DataFrame({
            "speaker": d["speaker"],
            "gold_text": d["text"],
            "label": d[LABEL_COL],
        })
        df["split"] = split_name
        return df

    train_df = to_df("train")
    test_df = to_df("test")

    full = pd.concat([train_df, test_df], ignore_index=True)
    full = full.dropna(subset=["label", "gold_text"]).copy()

    # Filter rare labels using combined train+test counts, same MIN_COUNT convention
    # as the Switchboard pipeline.
    vc = full["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    full = full[full["label"].isin(keep)].copy()

    full["rm_fillers_only"] = full["gold_text"].map(rm_fillers_only)
    full["clean_heavy"] = full["gold_text"].map(heavy_clean)

    full.to_csv(os.path.join(OUT_DIR, "utterances_mrda.csv"), index=False)

    train_df = full[full["split"] == "train"]
    test_df = full[full["split"] == "test"]

    print(f"Kept labels ({LABEL_COL}, min_count={MIN_COUNT}): {sorted(keep.tolist())}")
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")
    print(f"Meetings: 75 total (canonical MRDA split, meeting-disjoint by construction)")

    save_json = {
        "dataset": "wylupek/mrda-corpus",
        "label_col": LABEL_COL,
        "min_count": MIN_COUNT,
        "labels_kept": sorted(keep.tolist()),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "split_note": "Uses the dataset's own canonical train/test split (meeting-disjoint); "
                       "speaker codes are reused across meetings in this release so meeting id "
                       "cannot be recovered independently to build a custom split.",
    }
    with open(os.path.join(OUT_DIR, "mrda_split_info.json"), "w") as f:
        json.dump(save_json, f, indent=2)


if __name__ == "__main__":
    main()
