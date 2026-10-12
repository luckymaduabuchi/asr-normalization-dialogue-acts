#!/usr/bin/env bash
# Runs GPT-2 (decoder-only/causal attention) and BiLSTM (non-Transformer,
# non-attention, trained from scratch) on Switchboard, 9-condition sweep each,
# to test whether the RoBERTa/BERT/ALBERT recovery-from-filler-removal pattern
# is specific to bidirectional self-attention or holds more generally.
# Waits for the main model x dataset grid (scripts/orchestration/run_all_models_datasets.sh) to
# finish first, since there's one shared GPU.
set -e
cd /home/vm-user/Desktop/connl

echo "=== Waiting for main model x dataset grid to finish ==="
until grep -q "FULL GRID COMPLETE" run_all_models_datasets.log 2>/dev/null; do
  sleep 15
done
echo "=== Main grid done, proceeding ==="

echo "=== GPT-2 / switchboard ==="
python3 -u scripts/train/train_transformer_orchestrator.py gpt2 results/switchboard/gpt2 switchboard

echo "=== BiLSTM / switchboard ==="
python3 -u scripts/train/train_transformer_orchestrator.py bilstm results/switchboard/bilstm switchboard

echo "=== EXTRA ARCHITECTURES COMPLETE ==="
