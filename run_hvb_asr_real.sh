#!/usr/bin/env bash
# Runs the new asr_real condition (real Whisper ASR transcripts) through the
# 5 remaining models (RoBERTa, BERT, ALBERT, GPT-2, BiLSTM) on HVB only,
# reusing the existing per-model worker scripts directly (each is a single
# condition/model/dataset run with its own retry loop below, mirroring the
# orchestrator's subprocess-per-attempt pattern).
set -e
cd /home/vm-user/Desktop/connl

run_with_retry () {
  local desc="$1"; shift
  for attempt in 1 2 3 4 5; do
    echo "=== $desc: attempt $attempt/5 ==="
    if python3 -u "$@"; then
      echo "=== $desc: succeeded ==="
      return 0
    fi
    echo "=== $desc: failed, retrying in 30s ==="
    sleep 30
  done
  echo "=== $desc: giving up after 5 attempts ==="
  return 1
}

run_with_retry "RoBERTa/hvb/asr_real" train_transformer_worker.py asr_real roberta-base results_hvb_roberta hvb
run_with_retry "BERT/hvb/asr_real"    train_transformer_worker.py asr_real bert-base-cased results_hvb_bert hvb
run_with_retry "ALBERT/hvb/asr_real"  train_transformer_worker.py asr_real albert-base-v2 results_hvb_albert hvb
run_with_retry "GPT-2/hvb/asr_real"   train_transformer_worker.py asr_real gpt2 results_hvb_gpt2 hvb
run_with_retry "BiLSTM/hvb/asr_real"  train_bilstm_worker.py asr_real results_hvb_bilstm hvb

echo "=== HVB asr_real, all 5 models complete ==="
