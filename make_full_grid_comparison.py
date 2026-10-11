#!/usr/bin/env python3
"""
Regenerates full_grid_comparison.png: Macro-F1 for all 6 models, all 9
ablation conditions (same order as Table tab:full9x6x3), grouped by corpus.
HVB additionally gets a 10th bar, asr_real (real Whisper transcripts, same
train/test split as the other nine, Table tab:asrreal). MRDA's real-ASR
result is deliberately NOT plotted here: it comes from a different held-out
test set (all 75 ICSI meetings, independent of the wylupek/mrda-corpus split
used for the other 9 MRDA bars) and a model trained on gold_text only rather
than matched-condition asr_real training data, so it is not comparable in
height to its neighbors in this chart -- see Table tab:mrdaasrreal instead,
which reports it against its own matched gold baseline. Switchboard has no
real-ASR bar: its audio is LDC-licensed and unobtainable here.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ALL_RESULTS_CONSOLIDATED.csv")

MODELS = ["TF-IDF+LR", "RoBERTa", "BERT", "ALBERT", "GPT-2", "BiLSTM"]

# Same 9-condition order as Table tab:full9x6x3.
CORE_CONDITIONS = [
    "gold_text", "norm_colloq_only", "rm_punct_only", "rm_discourse_only",
    "gold_alpha_text", "rm_fillers_only", "clean_light", "clean_heavy", "asr_noise",
]
PANELS = [
    ("switchboard", "Switchboard (telephone calls)", CORE_CONDITIONS),
    ("mrda", "MRDA (meetings)", CORE_CONDITIONS),
    ("hvb", "SLUE-HVB (call-center)", CORE_CONDITIONS + ["asr_real"]),
]

cmap = plt.get_cmap("tab10")
COLORS = {cond: cmap(i) for i, cond in enumerate(CORE_CONDITIONS)}
COLORS["asr_real"] = "crimson"
LABELS = {cond: cond for cond in CORE_CONDITIONS}
LABELS["asr_real"] = "asr_real (Whisper)"

fig, axes = plt.subplots(1, 3, figsize=(26, 6.5))
fig.suptitle("Macro-F1 by condition across all 6 models and 3 datasets", fontsize=14)

for ax, (dataset, title, conditions) in zip(axes, PANELS):
    n_cond = len(conditions)
    x = np.arange(len(MODELS))
    width = 0.85 / n_cond
    for i, cond in enumerate(conditions):
        vals = []
        for model in MODELS:
            sub = df[(df.model == model) & (df.dataset == dataset) & (df.condition == cond)]
            vals.append(sub["macro_f1"].values[0] if len(sub) else np.nan)
        offset = (i - (n_cond - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=LABELS[cond], color=COLORS[cond], edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1")

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.08))

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig("full_grid_comparison.png", dpi=200, bbox_inches="tight")
print("Saved full_grid_comparison.png")
