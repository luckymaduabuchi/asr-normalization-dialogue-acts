#!/usr/bin/env bash
# Extends the full 9-condition Switchboard-style sweep to MRDA and HVB for the
# 5 remaining models (RoBERTa, BERT, ALBERT, GPT-2, BiLSTM). Each orchestrator
# call resumes from its existing metrics_summary.csv, so the 3 already-done
# conditions (gold_text, rm_fillers_only, clean_heavy) are skipped automatically
# and only the 6 new conditions run.
set -e
cd /home/vm-user/Desktop/connl

echo "=== RoBERTa / mrda (full 9) ==="
python3 -u train_transformer_orchestrator.py roberta-base results_mrda_roberta mrda

echo "=== BERT / mrda (full 9) ==="
python3 -u train_transformer_orchestrator.py bert-base-cased results_mrda_bert mrda

echo "=== ALBERT / mrda (full 9) ==="
python3 -u train_transformer_orchestrator.py albert-base-v2 results_mrda_albert mrda

echo "=== GPT-2 / mrda (full 9) ==="
python3 -u train_transformer_orchestrator.py gpt2 results_mrda_gpt2 mrda

echo "=== BiLSTM / mrda (full 9) ==="
python3 -u train_transformer_orchestrator.py bilstm results_mrda_bilstm mrda

echo "=== RoBERTa / hvb (full 9) ==="
python3 -u train_transformer_orchestrator.py roberta-base results_hvb_roberta hvb

echo "=== BERT / hvb (full 9) ==="
python3 -u train_transformer_orchestrator.py bert-base-cased results_hvb_bert hvb

echo "=== ALBERT / hvb (full 9) ==="
python3 -u train_transformer_orchestrator.py albert-base-v2 results_hvb_albert hvb

echo "=== GPT-2 / hvb (full 9) ==="
python3 -u train_transformer_orchestrator.py gpt2 results_hvb_gpt2 hvb

echo "=== BiLSTM / hvb (full 9) ==="
python3 -u train_transformer_orchestrator.py bilstm results_hvb_bilstm hvb

echo "=== FULL9 MRDA+HVB COMPLETE ==="
