# Superseded

The contents of this directory were produced by `bootstrap_ci.py`, which is no
longer used to generate any number in the paper.

Two problems, both fixed in `significance_tests.py` / `results_significance/`:

1. The conversation split used here (`bootstrap_ci.py`'s own
   `np.random.default_rng(42).shuffle`) differs from the split used everywhere
   else in the paper (`train_ablations.py`'s
   `sklearn.train_test_split(..., random_state=42)`, saved at
   `results_ablation/split_conversations.json`). That's why `gold_text` Macro-F1
   here (~0.289) doesn't match the ablation table or any other table in the
   paper (~0.311) -- it's a different train/test split, not a discrepancy in the
   model or data.

2. `significance_tests.csv` in this directory reports a "sign-style test"
   p-value that is identical across Macro-F1/Micro-F1/Accuracy for a given
   comparison. That's not a coincidence -- the p-value only depends on how many
   of the 300 bootstrap replicates were nonpositive (0/300 here for every
   metric), so it collapses to the same number regardless of metric. It is not
   a valid independent per-metric significance test.

See `results_significance/` for the corrected analysis: McNemar's test, a
conversation-level permutation test, and bootstrap CIs computed on the same
split used throughout the rest of the paper.
