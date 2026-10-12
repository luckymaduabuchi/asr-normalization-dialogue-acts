#!/usr/bin/env python3
"""
Computes McNemar's test (gold_text vs rm_fillers_only, gold_text vs clean_heavy)
for every model x dataset combination that has saved prediction arrays
(the 15 transformer/BiLSTM combos; TF-IDF+LR's 3 combos were already computed
by scripts/significance/significance_tests.py / scripts/train/train_mrda.py / scripts/train/train_hvb.py).
Writes results/cross_corpus/significance/full_grid_mcnemar.csv.
"""
import os
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

COMBOS = [
    ("RoBERTa", "switchboard", "results/switchboard/roberta"),
    ("RoBERTa", "mrda", "results/mrda/roberta"),
    ("RoBERTa", "hvb", "results/hvb/roberta"),
    ("BERT", "switchboard", "results/switchboard/bert"),
    ("BERT", "mrda", "results/mrda/bert"),
    ("BERT", "hvb", "results/hvb/bert"),
    ("ALBERT", "switchboard", "results/switchboard/albert"),
    ("ALBERT", "mrda", "results/mrda/albert"),
    ("ALBERT", "hvb", "results/hvb/albert"),
    ("GPT-2", "switchboard", "results/switchboard/gpt2"),
    ("GPT-2", "mrda", "results/mrda/gpt2"),
    ("GPT-2", "hvb", "results/hvb/gpt2"),
    ("BiLSTM", "switchboard", "results/switchboard/bilstm"),
    ("BiLSTM", "mrda", "results/mrda/bilstm"),
    ("BiLSTM", "hvb", "results/hvb/bilstm"),
]

rows = []
for model, dataset, out_dir in COMBOS:
    y_test = np.load(os.path.join(out_dir, "y_test_ids.npy"))
    base = np.load(os.path.join(out_dir, f"preds_gold_text.npy"))
    base_correct = base == y_test
    for cond in ["rm_fillers_only", "clean_heavy"]:
        pred_path = os.path.join(out_dir, f"preds_{cond}.npy")
        if not os.path.exists(pred_path):
            continue
        other = np.load(pred_path)
        other_correct = other == y_test
        both_correct = int(np.sum(base_correct & other_correct))
        base_only = int(np.sum(base_correct & ~other_correct))
        other_only = int(np.sum(~base_correct & other_correct))
        both_wrong = int(np.sum(~base_correct & ~other_correct))
        table = [[both_correct, base_only], [other_only, both_wrong]]
        use_exact = (base_only + other_only) < 5000
        result = mcnemar(table, exact=use_exact, correction=not use_exact)
        rows.append({
            "model": model, "dataset": dataset, "comparison": f"gold_text vs {cond}",
            "baseline_only_correct": base_only, "other_only_correct": other_only,
            "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "n_test": len(y_test),
        })
        print(f"{model:10s} {dataset:12s} gold vs {cond:16s} base_only={base_only:5d} "
              f"other_only={other_only:5d} p={result.pvalue:.4g}")

df = pd.DataFrame(rows)
os.makedirs("results/cross_corpus/significance", exist_ok=True)
df.to_csv("results/cross_corpus/significance/full_grid_mcnemar.csv", index=False)
print("\nSaved results/cross_corpus/significance/full_grid_mcnemar.csv", df.shape)
