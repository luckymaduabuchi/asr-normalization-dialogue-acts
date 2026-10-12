#!/usr/bin/env python3
"""
Builds the utterance-level dataset for the scoped real-ASR MRDA sub-experiment:
3 ICSI meetings (Bmr013, Bed002, Bro008), all channels, single-act rows only
(split '|' rows excluded -- can't cleanly attribute one audio span to multiple
sequential dialogue acts). Adds a stable utterance_id and sorts by start time
within each meeting so segment-cutting/transcription can proceed in order.
"""
import glob
import os

import pandas as pd

from mrda_icsi_parse import parse_dialogue_acts

ROOT = "/tmp/icsi_check/icsi_core/ICSI"
MEETINGS = ['Bdb001', 'Bed002', 'Bed003', 'Bed004', 'Bed005', 'Bed006', 'Bed008', 'Bed009', 'Bed010',
            'Bed011', 'Bed012', 'Bed013', 'Bed014', 'Bed015', 'Bed016', 'Bed017', 'Bmr001', 'Bmr002',
            'Bmr003', 'Bmr005', 'Bmr006', 'Bmr007', 'Bmr008', 'Bmr009', 'Bmr010', 'Bmr011', 'Bmr012',
            'Bmr013', 'Bmr014', 'Bmr015', 'Bmr016', 'Bmr018', 'Bmr019', 'Bmr020', 'Bmr021', 'Bmr022',
            'Bmr023', 'Bmr024', 'Bmr025', 'Bmr026', 'Bmr027', 'Bmr028', 'Bmr029', 'Bmr030', 'Bmr031',
            'Bns001', 'Bns002', 'Bns003', 'Bro003', 'Bro004', 'Bro005', 'Bro007', 'Bro008', 'Bro010',
            'Bro011', 'Bro012', 'Bro013', 'Bro014', 'Bro015', 'Bro016', 'Bro017', 'Bro018', 'Bro019',
            'Bro021', 'Bro022', 'Bro023', 'Bro024', 'Bro025', 'Bro026', 'Bro027', 'Bro028', 'Bsr001',
            'Btr001', 'Btr002', 'Buw001']  # all 75 ICSI meetings

rows = []
for meeting in MEETINGS:
    da_files = sorted(glob.glob(os.path.join(ROOT, "DialogueActs", f"{meeting}.*.dialogue-acts.xml")))
    for da_path in da_files:
        channel = os.path.basename(da_path).split(".")[1]
        words_path = os.path.join(ROOT, "Words", f"{meeting}.{channel}.words.xml")
        if not os.path.exists(words_path):
            continue
        recs = parse_dialogue_acts(da_path, words_path)
        for r in recs:
            if r["start"] is None or r["end"] is None:
                continue
            if r["end"] <= r["start"]:
                continue
            rows.append({
                "meeting": meeting,
                "channel": channel,
                "participant": r["participant"],
                "start": r["start"],
                "end": r["end"],
                "duration": r["end"] - r["start"],
                "raw_type": r["raw_type"],
                "n_split_acts": r["n_split_acts"],
                "general_da": r["general_da"],
                "gold_text": r["text"],
            })

df = pd.DataFrame(rows)
n_total = len(df)

# Exclude multi-act split rows (n_split_acts > 1) -- can't cleanly segment/attribute audio.
df_single = df[df["n_split_acts"] == 1].copy()
n_excluded = n_total - len(df_single)

# Sanity filter: a handful of ICSI word-alignment spans have implausible gaps
# (e.g. a 6-word utterance spanning 302s), almost certainly annotation-timing
# artifacts rather than real speech tempo. Drop rows below 0.8 words/sec --
# no fluent speaker sustains that rate -- so segment-cutting doesn't hand
# Whisper multi-minute clips dominated by silence/cross-talk for a single label.
n_words = df_single["gold_text"].str.split().apply(len)
wps = n_words / df_single["duration"]
n_bad_timing = int((wps < 0.8).sum())
df_single = df_single[wps >= 0.8].copy()

df_single = df_single.sort_values(["meeting", "start"]).reset_index(drop=True)
df_single["utterance_id"] = [
    f"{row.meeting}_{row.channel}_{i:04d}" for i, row in enumerate(df_single.itertuples())
]

out_path = "/home/vm-user/Desktop/connl/results_mrda_real/scoped_utterances.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_single.to_csv(out_path, index=False)

print(f"Total single+split DA rows parsed: {n_total}")
print(f"Excluded split-act rows: {n_excluded} ({100*n_excluded/n_total:.1f}%)")
print(f"Excluded implausible-timing rows (<0.8 words/sec): {n_bad_timing}")
print(f"Final scoped dataset: {len(df_single)} rows -> {out_path}")
print()
print("Per meeting:")
print(df_single.groupby("meeting").size())
print()
print("Label distribution:")
print(df_single["general_da"].value_counts())
print()
print("Duration stats (seconds):")
print(df_single["duration"].describe())
