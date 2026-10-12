# ASR-style transcript normalization and dialogue act classification

Code and results for a study of how transcript normalization (filler removal,
punctuation removal, colloquial normalization, lowercasing) affects dialogue-act
classification across six classifiers (TF-IDF + logistic regression, RoBERTa,
BERT, ALBERT, GPT-2, BiLSTM) and three corpora (Switchboard, ICSI MRDA, SLUE-HVB).

All scripts are invoked from the repository root (e.g. `python
scripts/train/train_mrda.py`) — their input/output paths are relative to the
root, not to the script's own location.

## Layout

```
scripts/
  data_prep/      builds each normalization condition per corpus
  train/          TF-IDF+LR and neural training, incl. multi-seed orchestration
  real_asr/       Whisper transcription, WER, and the MRDA meeting-disjoint split
  significance/   McNemar tests, cluster bootstrap/permutation tests
  figures/        confusion matrices, comparison bars, the delta heatmap
  orchestration/  shell scripts that sequence the above across models/corpora
  legacy/         superseded scripts, kept for history (see legacy/README.md)

results/
  switchboard/    tfidf, roberta, bert, albert, gpt2, bilstm, multiseed
  mrda/           tfidf, roberta, bert, albert, gpt2, bilstm, multiseed,
                  meetingsplit, meetingsplit_conditions (real-ASR split)
  hvb/            tfidf, roberta, bert, albert, gpt2, bilstm, multiseed
  cross_corpus/   significance*, controls (scripts/data_prep/reproduce_paper_controls.py output),
                  summary (consolidated CSVs/figures spanning all three corpora)
  legacy/         superseded results, kept for history (see legacy/README.md)
```

Every `results/<corpus>/<model>/` directory holds `metrics_summary.csv`, the
test labels (`y_test_ids.npy`), and per-condition predictions
(`preds_<condition>.npy`), so every reported metric and statistical test can
be recomputed without retraining.

## What is in this repository

- **Data preparation** (`scripts/data_prep/`): `scripts/data_prep/make_text_conditions.py`,
  `scripts/data_prep/make_ablations.py` (Switchboard); `scripts/data_prep/mrda_prepare.py`, `scripts/data_prep/hvb_prepare.py`,
  `scripts/data_prep/extend_conditions_mrda_hvb.py` (MRDA, HVB); `scripts/data_prep/make_strict_filler_conditions.py`,
  `scripts/data_prep/make_pragmatic_norm_condition.py`, `scripts/data_prep/make_asr_noise_condition.py` (extra
  conditions).
- **Training** (`scripts/train/`): `scripts/train/train_ablations.py`, `scripts/train/train_mrda.py`,
  `scripts/train/train_hvb.py` (TF-IDF + LR); `scripts/train/train_transformer_worker.py`,
  `scripts/train/train_bilstm_worker.py` (neural models); `scripts/train/train_multiseed_orchestrator.py`
  (seeds 1-4; seed 42 is the main grid).
- **Real ASR** (`scripts/real_asr/`): `scripts/real_asr/hvb_generate_real_asr.py`,
  `scripts/real_asr/mrda_generate_real_asr.py` (Whisper-base.en), `scripts/real_asr/compute_wer_summary.py`,
  `scripts/real_asr/mrda_meeting_split.py`, `scripts/real_asr/mrda_meetingdisjoint_worker.py`.
- **Statistics** (`scripts/significance/`): `significance_tests*.py`,
  `scripts/significance/full_grid_significance.py` (McNemar, cluster-level permutation tests,
  cluster bootstrap intervals).
- **Reproducibility controls** (`scripts/data_prep/reproduce_paper_controls.py`):
  regenerates the punctuation-aware TF-IDF experiment, filler/backchannel
  counts, and the Switchboard extra-discourse-token control, writing to
  `results/cross_corpus/controls/`.
- **Figures** (`scripts/figures/`): confusion matrices, condition-comparison
  bars, and the cross-corpus delta heatmap.

Any directory containing a `SUPERSEDED.md` — `results/legacy/` — holds
earlier results that are not used in the paper; see that file for why each
is kept. The compiled manuscript is `paper/asilomar_2026_paper.pdf`.

## What is not included

- Raw corpora and audio. Switchboard comes from ConvoKit
  (`switchboard-processed-corpus`), MRDA from `wylupek/mrda-corpus` on HuggingFace
  and the ICSI Meeting Corpus (NXT XML and audio), SLUE-HVB from
  `asapp/slue-phase-2` (`hvb` config).
- Utterance tables with gold transcript text. Run the preparation scripts above to
  rebuild them. Whisper transcripts (`asr_real*.csv`) are included.
- Trained model weights and training logs.

## Environment

Python 3.9, torch 2.5.1, transformers 4.45.2, scikit-learn 1.5.1,
datasets 2.19.1, statsmodels 0.14.2, jiwer 4.0.0.

Some scripts contain absolute paths from the original machine and need adjusting.
