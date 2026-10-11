#!/bin/bash
# Chains: (1) wait for the MRDA meeting-disjoint orchestrator (already running,
# PID given as $1) to finish so we don't compete for the same shared GPU, then
# (2) runs rm_fillers_strict / clean_heavy_strict / pragmatic_norm across all
# 3 corpora x 5 non-TF-IDF models (TF-IDF already done, CPU-only, fast).
set -uo pipefail

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "Waiting for PID $WAIT_PID (MRDA meeting-disjoint orchestrator) to finish..."
  tail --pid="$WAIT_PID" -f /dev/null
  echo "MRDA meeting-disjoint orchestrator finished. Starting extra-conditions GPU sweep..."
fi

ENTRIES=(
  "roberta-base:results_transformer:switchboard"
  "bert-base-cased:results_transformer_bert:switchboard"
  "albert-base-v2:results_transformer_albert:switchboard"
  "gpt2:results_transformer_gpt2:switchboard"
  "bilstm:results_bilstm:switchboard"
  "roberta-base:results_mrda_roberta:mrda"
  "bert-base-cased:results_mrda_bert:mrda"
  "albert-base-v2:results_mrda_albert:mrda"
  "gpt2:results_mrda_gpt2:mrda"
  "bilstm:results_mrda_bilstm:mrda"
  "roberta-base:results_hvb_roberta:hvb"
  "bert-base-cased:results_hvb_bert:hvb"
  "albert-base-v2:results_hvb_albert:hvb"
  "gpt2:results_hvb_gpt2:hvb"
  "bilstm:results_hvb_bilstm:hvb"
)

for entry in "${ENTRIES[@]}"; do
  IFS=":" read -r model outdir dataset <<< "$entry"
  if [ "$dataset" = "mrda" ]; then
    conds="rm_fillers_strict,clean_heavy_strict"
  else
    conds="rm_fillers_strict,clean_heavy_strict,pragmatic_norm"
  fi
  echo ""
  echo "=== $dataset / $model / $conds ==="
  python3 -u train_extra_conditions_orchestrator.py "$model" "$outdir" "$dataset" "$conds"
done

echo ""
echo "ALL EXTRA-CONDITIONS GPU RUNS COMPLETE"
