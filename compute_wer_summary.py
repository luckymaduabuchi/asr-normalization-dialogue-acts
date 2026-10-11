#!/usr/bin/env python3
"""
Reproducible WER artifact for the real-ASR claims in the paper (HVB 18.3%
test-split WER, MRDA 41.0% meeting-disjoint-test-partition WER). Computes
word error rate with jiwer, comparing each Whisper hypothesis against its
gold_text reference under an explicit normalization transform (lowercase,
strip punctuation, collapse whitespace -- NOT jiwer's default behavior;
jiwer.wer() with no transform arguments compares raw strings, which
inflates WER by counting Whisper's punctuated/cased output as mismatched
against gold_text's lowercase, comma-tokenized format even where the
words recognized are identical. The paper's originally reported 54.1%
(HVB) / 59.1% (MRDA) figures came from exactly this faulty method: an
earlier, undocumented computation that never applied text normalization
before comparing hypothesis to reference, so it was measuring formatting
mismatch rather than genuine recognition error. Those figures were
incorrect and have been replaced throughout the paper with the values
this script produces.)

HVB: matches on ['orig_idx','split'] (orig_idx resets per split, so a plain
merge on orig_idx alone silently cartesian-joins train and test rows --
the bug fixed earlier this session). Reports train, test, and combined.

MRDA: reports the full 100,513-utterance transcribed pool (matches the
Section IV headline figure) and the 21,350-utterance meeting-disjoint test
partition actually evaluated in Table V, so the two numbers cited in the
paper (pool-level and evaluation-level) are both traceable to a saved file.

Output: wer_summary.csv (corpus, split, n_utterances, wer_pct, hits,
substitutions, deletions, insertions).
"""
import json
import pandas as pd
import jiwer

NORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

rows = []


def score(corpus, split, ref, hyp):
    ref = ref.fillna("").astype(str).tolist()
    hyp = hyp.fillna("").astype(str).tolist()
    out = jiwer.process_words(ref, hyp, reference_transform=NORM, hypothesis_transform=NORM)
    rows.append({
        "corpus": corpus, "split": split, "n_utterances": len(ref),
        "reference_normalization": "lowercase+strip_punct+collapse_whitespace",
        "wer_pct": round(100 * out.wer, 2),
        "hits": out.hits, "substitutions": out.substitutions,
        "deletions": out.deletions, "insertions": out.insertions,
    })
    print(f"{corpus}/{split}: n={len(ref)} WER={100*out.wer:.2f}% "
          f"(hits={out.hits}, sub={out.substitutions}, del={out.deletions}, ins={out.insertions})")


# ---- HVB ----
idx = pd.read_csv("results_hvb/hvb_filtered_index.csv")
train_idx = idx[idx.split == "train"].copy()
test_idx = idx[idx.split == "test"].copy()
train_idx["split_tag"] = "train"
test_idx["split_tag"] = "test"

train_asr = pd.read_csv("results_hvb/asr_real_train.csv")
test_asr = pd.read_csv("results_hvb/asr_real_test.csv")

train_m = train_idx.merge(train_asr, on="orig_idx", how="inner")
test_m = test_idx.merge(test_asr, on="orig_idx", how="inner")

assert len(train_m) == len(train_idx), f"HVB train merge mismatch: {len(train_m)} vs {len(train_idx)}"
assert len(test_m) == len(test_idx), f"HVB test merge mismatch: {len(test_m)} vs {len(test_idx)}"

score("HVB", "train", train_m["gold_text"], train_m["asr_real"])
score("HVB", "test", test_m["gold_text"], test_m["asr_real"])
combined_ref = pd.concat([train_m["gold_text"], test_m["gold_text"]], ignore_index=True)
combined_hyp = pd.concat([train_m["asr_real"], test_m["asr_real"]], ignore_index=True)
score("HVB", "train+test (combined)", combined_ref, combined_hyp)

# ---- MRDA ----
scoped = pd.read_csv("results_mrda_real/scoped_utterances.csv")
asr = pd.read_csv("results_mrda_real/asr_real.csv")
pool = scoped.merge(asr, on="utterance_id", how="inner")
score("MRDA", "full 75-meeting pool", pool["gold_text"], pool["asr_real"])

split = json.load(open("results_mrda_meetingsplit/split_meetings.json"))
test_meetings = set(split["test_meetings"])
test_pool = pool[pool["meeting"].isin(test_meetings)]
score("MRDA", "meeting-disjoint test partition (15 meetings)", test_pool["gold_text"], test_pool["asr_real"])

pd.DataFrame(rows).to_csv("wer_summary.csv", index=False)
print("\nSaved wer_summary.csv")
