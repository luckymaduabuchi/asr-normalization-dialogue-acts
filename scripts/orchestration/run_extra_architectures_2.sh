#!/usr/bin/env bash
# Extends GPT-2 and BiLSTM to MRDA and HVB (3 core conditions each), completing
# the full 6-model x 3-dataset grid. Waits for scripts/orchestration/run_extra_architectures.sh
# (GPT-2 + BiLSTM on Switchboard) to finish first, since there's one shared GPU.
set -e
cd /home/vm-user/Desktop/connl

echo "=== Waiting for GPT-2/BiLSTM on switchboard to finish ==="
until grep -q "EXTRA ARCHITECTURES COMPLETE" run_extra_architectures.log 2>/dev/null; do
  sleep 15
done
echo "=== Proceeding with MRDA/HVB for GPT-2 and BiLSTM ==="

echo "=== GPT-2 / mrda ==="
python3 -u scripts/train/train_transformer_orchestrator.py gpt2 results/mrda/gpt2 mrda

echo "=== GPT-2 / hvb ==="
python3 -u scripts/train/train_transformer_orchestrator.py gpt2 results/hvb/gpt2 hvb

echo "=== BiLSTM / mrda ==="
python3 -u scripts/train/train_transformer_orchestrator.py bilstm results/mrda/bilstm mrda

echo "=== BiLSTM / hvb ==="
python3 -u scripts/train/train_transformer_orchestrator.py bilstm results/hvb/bilstm hvb

echo "=== FULL 6x3 GRID COMPLETE ==="
