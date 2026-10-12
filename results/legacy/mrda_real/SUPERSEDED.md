# Status of files in this directory

This directory mixes two things: source data that is still authoritative,
and an earlier real-ASR result that has been superseded by
`results/mrda/meetingsplit/`. They are not separated into different
directories because the newer experiment reuses the source data below.

## Still authoritative (reused by results/mrda/meetingsplit/)
- `scoped_utterances.csv` -- the 100,513-utterance, 75-meeting ICSI-NXT-XML
  pool (gold_text + general_da labels + meeting field).
- `asr_real.csv` -- Whisper-base.en transcripts for that pool.

## Superseded -- do not cite these numbers in the paper
Produced by the exact-text-deduplication approach (train on
`wylupek/mrda-corpus`, dedupe the ICSI-NXT-XML test pool against it).
Discarded after discovering the training split's lack of a meeting
identifier meant deduplication could not guarantee meeting-level
independence (51.2% of the pool was found verbatim in that training
split). Replaced by the meeting-disjoint split in
`results/mrda/meetingsplit/` (self-built train/test partition on this same
pool, split by the `meeting` field directly, models trained from scratch
on the train partition). The paper's Table V and all MRDA real-ASR prose
report the `results/mrda/meetingsplit/` numbers only.

Superseded files: `metrics_summary.csv`, `metrics_summary_dedup.csv`,
`mcnemar_tests.csv`, `mcnemar_dedup.csv`, `dedup_mask.npy`,
`test_df_ordered.csv`, `preds_*_{gold,asr}.npy`, `y_test_ids.npy`,
`label2id.json`. Kept for the paper's own methodological narrative
(Section IV, "Real ASR evaluation on MRDA") documenting why the approach
was discarded, not as a source of reported figures.
