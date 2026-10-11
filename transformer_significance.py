#!/usr/bin/env python3
"""
McNemar's test on the RoBERTa predictions saved by train_transformer_worker.py,
mirroring significance_tests.py's treatment of the TF-IDF+LR baseline, for the
same gold_text vs rm_fillers_only / clean_heavy comparisons.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

OUT_DIR = "results_transformer"
BASELINE = "gold_text"
COMPARISONS = ["rm_fillers_only", "clean_heavy"]


def main():
    y_test = np.load(f"{OUT_DIR}/y_test_ids.npy")
    base = np.load(f"{OUT_DIR}/preds_{BASELINE}.npy")
    base_correct = base == y_test

    rows = []
    for cond in COMPARISONS:
        other = np.load(f"{OUT_DIR}/preds_{cond}.npy")
        other_correct = other == y_test

        both_correct = int(np.sum(base_correct & other_correct))
        base_only = int(np.sum(base_correct & ~other_correct))
        other_only = int(np.sum(~base_correct & other_correct))
        both_wrong = int(np.sum(~base_correct & ~other_correct))

        table = [[both_correct, base_only], [other_only, both_wrong]]
        use_exact = (base_only + other_only) < 5000
        result = mcnemar(table, exact=use_exact, correction=not use_exact)

        rows.append({
            "comparison": f"{BASELINE} vs {cond}",
            "both_correct": both_correct,
            "baseline_only_correct": base_only,
            "other_only_correct": other_only,
            "both_wrong": both_wrong,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "exact": use_exact,
        })
        print(f"{cond}: baseline_only={base_only} other_only={other_only} "
              f"stat={result.statistic} p={result.pvalue:.6g}")

    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/mcnemar_tests.csv", index=False)


if __name__ == "__main__":
    main()
