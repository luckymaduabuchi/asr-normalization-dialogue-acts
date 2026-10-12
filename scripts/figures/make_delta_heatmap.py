#!/usr/bin/env python3
"""
Regenerates results/cross_corpus/summary/delta_heatmap.png (Macro-F1 delta, gold_text - rm_fillers_only,
6 models x 3 corpora) from the current, corrected values in Table tab:grid.
The previous copy of this figure was produced by an untracked one-off script
during an earlier session and went stale after the BiLSTM empty-sequence
pooling bug was fixed (BiLSTM/MRDA's true delta is 0.052, not the 0.041 the
old image showed); this script makes the figure regeneratable going forward.
"""
import numpy as np
import matplotlib.pyplot as plt

MODELS = ["TF-IDF+LR", "RoBERTa", "BERT", "ALBERT", "GPT-2", "BiLSTM"]
CORPORA = ["Switchboard", "MRDA", "HVB"]

# Delta = gold_text - rm_fillers_only, matching Table tab:grid exactly (post BiLSTM fix)
DELTA = np.array([
    [0.011, 0.045, 0.053],   # TF-IDF+LR
    [-0.003, 0.036, 0.045],  # RoBERTa
    [0.001, 0.028, 0.046],   # BERT
    [0.037, 0.047, 0.043],   # ALBERT
    [-0.006, 0.049, 0.001],  # GPT-2
    [0.015, 0.052, 0.048],   # BiLSTM (corrected: MRDA 0.041 -> 0.052)
])

fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(DELTA, cmap="RdBu_r", vmin=-0.02, vmax=0.13, aspect="auto")

ax.set_xticks(np.arange(len(CORPORA)))
ax.set_xticklabels(CORPORA)
ax.set_yticks(np.arange(len(MODELS)))
ax.set_yticklabels(MODELS)

for i in range(len(MODELS)):
    for j in range(len(CORPORA)):
        ax.text(j, i, f"{DELTA[i, j]:.3f}", ha="center", va="center", fontsize=11)

ax.set_title("Macro-F1 delta (gold_text - rm_fillers_only)\nnegative = filler removal helped")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Delta (positive = filler removal hurt)")

fig.tight_layout()
fig.savefig("results/cross_corpus/summary/delta_heatmap.png", dpi=200)
fig.savefig("IEEE_Conference_Template__3_/delta_heatmap.png", dpi=200)
print("Saved results/cross_corpus/summary/delta_heatmap.png (both copies)")
