# ASR-style transcript normalization and dialogue act classification

Code and results for a study of how transcript normalization (filler removal,
punctuation removal, colloquial normalization, lowercasing) affects dialogue-act
classification across six classifiers (TF-IDF + logistic regression, RoBERTa,
BERT, ALBERT, GPT-2, BiLSTM) and three corpora (Switchboard, ICSI MRDA, SLUE-HVB).

## What is in this repository

- **Data preparation:** `make_text_conditions.py`, `make_ablations.py` (Switchboard);
  `mrda_prepare.py`, `hvb_prepare.py`, `extend_conditions_mrda_hvb.py` (MRDA, HVB);
  `make_strict_filler_conditions.py`, `make_pragmatic_norm_condition.py`,
  `make_asr_noise_condition.py` (extra conditions).
- **Training:** `train_ablations.py`, `train_mrda.py`, `train_hvb.py` (TF-IDF + LR);
  `train_transformer_worker.py`, `train_bilstm_worker.py` (neural models);
  `train_multiseed_orchestrator.py` (seeds 1-4; seed 42 is the main grid).
- **Real ASR:** `hvb_generate_real_asr.py`, `mrda_generate_real_asr.py`
  (Whisper-base.en), `compute_wer_summary.py`, `mrda_meeting_split.py`,
  `mrda_meetingdisjoint_worker.py`.
- **Statistics:** `significance_tests*.py`, `full_grid_significance.py`
  (McNemar, cluster-level permutation tests, cluster bootstrap intervals).
- **Reproducibility controls:** `reproduce_paper_controls.py` regenerates the
  punctuation-aware TF-IDF experiment, filler/backchannel counts, and the
  Switchboard extra-discourse-token control in `results_controls/`.
- **Results:** every `results_*/` directory holds `metrics_summary.csv`, the test
  labels (`y_test_ids.npy`) and per-condition predictions (`preds_<condition>.npy`),
  so all reported metrics and tests can be recomputed without retraining.
  `results_multiseed*/` hold the additional seeds.

Directories containing a `SUPERSEDED.md` hold earlier results that are not used.
The manuscript source and bibliography are in `IEEE_Conference_Template__3_/`.

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
