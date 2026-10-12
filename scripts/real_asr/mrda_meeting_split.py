#!/usr/bin/env python3
"""
Builds a genuinely meeting-disjoint train/test split for MRDA real-ASR
evaluation, replacing the exact-text-deduplication workaround (which removed
verbatim training duplicates but could not guarantee meeting-level
independence, since wylupek/mrda-corpus carries no meeting identifier).

Uses results/legacy/mrda_real/scoped_utterances.csv directly: this is the
ICSI-NXT-XML-derived dataset spanning all 75 meetings, built independently
of wylupek/mrda-corpus, and it DOES carry a meeting column. Splitting on
that column and training fresh models on the train-meeting partition makes
the resulting comparison self-consistently meeting-disjoint by construction
-- no dependence on (or reconciliation with) the wylupek split at all.

Target ~80/20 utterance-level split via greedy bin-packing over meetings
(meetings vary widely in utterance count, so a naive random meeting shuffle
would risk a very unbalanced test-set size).
"""
import json
import numpy as np
import pandas as pd

SCOPED_CSV = "results/legacy/mrda_real/scoped_utterances.csv"
OUT_PATH = "results/mrda/meetingsplit/split_meetings.json"
TEST_FRACTION = 0.20
SEED = 42

import os
os.makedirs("results/mrda/meetingsplit", exist_ok=True)

df = pd.read_csv(SCOPED_CSV)
counts = df.groupby("meeting").size().sort_values(ascending=False)
total = counts.sum()
target_test = total * TEST_FRACTION

rng = np.random.default_rng(SEED)
meetings = counts.index.tolist()
# Shuffle within similar-size bands so the greedy pick isn't fully deterministic
order = rng.permutation(len(meetings))
meetings_shuffled = [meetings[i] for i in order]

test_meetings, test_n = [], 0
for m in meetings_shuffled:
    if test_n >= target_test:
        break
    test_meetings.append(m)
    test_n += counts[m]

train_meetings = [m for m in meetings if m not in set(test_meetings)]
train_n = counts[train_meetings].sum()

assert set(train_meetings).isdisjoint(set(test_meetings))
assert len(train_meetings) + len(test_meetings) == len(meetings)

split = {
    "seed": SEED,
    "test_fraction_target": TEST_FRACTION,
    "train_meetings": sorted(train_meetings),
    "test_meetings": sorted(test_meetings),
    "train_utterances": int(train_n),
    "test_utterances": int(test_n),
}
with open(OUT_PATH, "w") as f:
    json.dump(split, f, indent=2)

print(f"Meetings: {len(meetings)} total -> {len(train_meetings)} train / {len(test_meetings)} test")
print(f"Utterances: {train_n} train ({train_n/total:.1%}) / {test_n} test ({test_n/total:.1%})")
print(f"Saved split to {OUT_PATH}")

# Label balance sanity check
label_counts = df.merge(pd.DataFrame({"meeting": train_meetings, "split": "train"}), on="meeting", how="left")
label_counts["split"] = label_counts["split"].fillna("test")
print("\nLabel distribution by split (fraction of each label's utterances in test):")
tbl = df.assign(split=np.where(df["meeting"].isin(test_meetings), "test", "train"))
frac_test = tbl.groupby("general_da")["split"].apply(lambda s: (s == "test").mean())
print(frac_test.sort_values(ascending=False).to_string())
