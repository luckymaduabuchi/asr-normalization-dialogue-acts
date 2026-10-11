#!/usr/bin/env bash
# Runs the full model x dataset grid sequentially (one GPU, one job at a time).
# BERT-on-Switchboard is already running externally when this is launched; this
# script waits for it to finish, then runs the remaining 7 combinations in order:
#   ALBERT/switchboard (9 conditions)
#   RoBERTa/mrda, BERT/mrda, ALBERT/mrda (3 core conditions each)
#   RoBERTa/hvb, BERT/hvb, ALBERT/hvb (3 core conditions each)
set -e
cd /home/vm-user/Desktop/connl

echo "=== Waiting for BERT/switchboard to finish ==="
until grep -q "All conditions processed" results_transformer_bert_run.log 2>/dev/null; do
  sleep 15
done
echo "=== BERT/switchboard done, proceeding ==="

echo "=== ALBERT / switchboard ==="
python3 -u train_transformer_orchestrator.py albert-base-v2 results_transformer_albert switchboard

echo "=== RoBERTa / mrda ==="
python3 -u train_transformer_orchestrator.py roberta-base results_mrda_roberta mrda

echo "=== BERT / mrda ==="
python3 -u train_transformer_orchestrator.py bert-base-cased results_mrda_bert mrda

echo "=== ALBERT / mrda ==="
python3 -u train_transformer_orchestrator.py albert-base-v2 results_mrda_albert mrda

echo "=== RoBERTa / hvb ==="
python3 -u train_transformer_orchestrator.py roberta-base results_hvb_roberta hvb

echo "=== BERT / hvb ==="
python3 -u train_transformer_orchestrator.py bert-base-cased results_hvb_bert hvb

echo "=== ALBERT / hvb ==="
python3 -u train_transformer_orchestrator.py albert-base-v2 results_hvb_albert hvb

echo "=== FULL GRID COMPLETE ==="
