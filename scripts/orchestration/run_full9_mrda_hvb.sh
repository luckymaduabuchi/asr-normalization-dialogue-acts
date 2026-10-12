#!/usr/bin/env bash
# Extends the full 9-condition Switchboard-style sweep to MRDA and HVB for the
# 5 remaining models (RoBERTa, BERT, ALBERT, GPT-2, BiLSTM). Each orchestrator
# call resumes from its existing metrics_summary.csv, so the 3 already-done
# conditions (gold_text, rm_fillers_only, clean_heavy) are skipped automatically
# and only the 6 new conditions run.
set -e
cd /home/vm-user/Desktop/connl

echo "=== RoBERTa / mrda (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py roberta-base results/mrda/roberta mrda

echo "=== BERT / mrda (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py bert-base-cased results/mrda/bert mrda

echo "=== ALBERT / mrda (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py albert-base-v2 results/mrda/albert mrda

echo "=== GPT-2 / mrda (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py gpt2 results/mrda/gpt2 mrda

echo "=== BiLSTM / mrda (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py bilstm results/mrda/bilstm mrda

echo "=== RoBERTa / hvb (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py roberta-base results/hvb/roberta hvb

echo "=== BERT / hvb (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py bert-base-cased results/hvb/bert hvb

echo "=== ALBERT / hvb (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py albert-base-v2 results/hvb/albert hvb

echo "=== GPT-2 / hvb (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py gpt2 results/hvb/gpt2 hvb

echo "=== BiLSTM / hvb (full 9) ==="
python3 -u scripts/train/train_transformer_orchestrator.py bilstm results/hvb/bilstm hvb

echo "=== FULL9 MRDA+HVB COMPLETE ==="
