#!/usr/bin/env python3
"""
Regenerates the experimental-pipeline diagram (image.png), fixing one
inaccurate claim in the prior version: "Same training data across all nine
conditions" is wrong -- each condition trains AND tests on its own
transformed text (that's the point of the ablation). What's actually held
constant is the row-level train/test split, not the text itself.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(15.5, 10.2))
ax.set_xlim(0, 155)
ax.set_ylim(0, 102)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec="black", fontsize=8.3, weight="normal", ha="center", lw=1.1, boxstyle="round,pad=0.012,rounding_size=0.5"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va="center", fontsize=fontsize, weight=weight,
             zorder=3, wrap=True, linespacing=1.35)


def arrow(x0, y0, x1, y1, color="black"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=color), zorder=1)


ax.text(77.5, 99.5, "Experimental pipeline for transcript normalization ablations",
        ha="center", va="center", fontsize=15, weight="bold")

# Row 1: corpora
corp_y, corp_h = 87, 9.5
box(3, corp_y, 40, corp_h,
    "Switchboard dialogue-act transcripts\nOriginal utterance text + DAMSL-style labels\n(Telephone conversations, 27 labels)",
    fc="#dbe9fb")
box(46, corp_y, 40, corp_h,
    "MRDA dialogue-act transcripts\nOriginal utterance text + DAMSL-style labels\n(Meetings, 12 labels)",
    fc="#dbe9fb")
box(89, corp_y, 40, corp_h,
    "SLUE-HVB dialogue-act transcripts\nOriginal utterance text + DAMSL-style labels\n(Call-center, 16 labels)",
    fc="#dbe9fb")
for cx in (23, 66, 109):
    arrow(cx, corp_y, cx, corp_y - 2.3)

# Conditions band
cond_y, cond_h = 71.5, 13.5
ax.add_patch(FancyBboxPatch((3, cond_y - 1.2), 126, cond_h + 1.5, boxstyle="round,pad=0.02", fc="none", ec="#3355aa", lw=1.2, ls="--", zorder=1))
ax.text(66, cond_y + cond_h - 0.3, "Transcript conditions: 9 shared conditions + asr\\_real for HVB and MRDA",
        ha="center", va="center", fontsize=9.5, weight="bold", color="#20335e")

cond_labels = [
    ("gold_text", "Original\nprocessed transcript text"),
    ("gold_alpha_text", "Alphanumeric-only\nversion (alpha filter)"),
    ("rm_punct_only", "Remove all\npunctuation"),
    ("norm_colloq_only", "Normalize colloquials\n(gonna→going to,\nwanna→want to)"),
    ("rm_discourse_only", "Remove discourse\nmarkers (you know,\nI mean, well, ...)"),
    ("rm_fillers_only", "Remove filled pauses\n(um, uh, erm, ah, ...)"),
    ("clean_light", "Lowercase + remove\npunctuation (light\ncleaning)"),
    ("clean_heavy", "Lowercase + remove\npunct. + normalize\ncolloq. + remove fillers"),
    ("asr_noise", "Synthetic ASR noise\ninjection (≈20% WER\nat word level)"),
]
n = len(cond_labels)
cw = 12.4
gap = 0.55
start_x = 3.6
for i, (name, desc) in enumerate(cond_labels):
    x = start_x + i * (cw + gap)
    box(x, cond_y, cw, cond_h - 1.5, f"{name}\n\n{desc}", fc="white", fontsize=6.7)

box(131, cond_y - 1.2, 21, cond_h + 1.5,
    "asr_real\n(HVB and MRDA only)\n\nWhisper-base.en\nhypotheses\n\n"
    "HVB: full train and\ntest sets\n\n"
    "MRDA: self-built\nmeeting-disjoint\nsplit (60/15\nmeetings), models\ntrained from scratch\n\n"
    "Switchboard:\nunavailable",
    fc="#e7f4e7", ec="#2a7a2a", fontsize=6.4)

arrow(66, cond_y - 1.2, 66, cond_y - 3.3)

# Splits
split_y, split_h = 62.5, 6.5
box(3, split_y, 126, split_h,
    "Fixed corpus-specific train/test splits\n"
    "Switchboard: conversation-level 80/20   |   MRDA: canonical meeting-disjoint split   |   SLUE-HVB: canonical issue-disjoint split",
    fc="#eaf1fc", fontsize=8.5)
arrow(66, split_y, 66, split_y - 2.3)

# Feature extraction
feat_y, feat_h = 54.5, 6.5
box(3, feat_y, 126, feat_h,
    "Feature extraction / Tokenization (within each condition × corpus)\n"
    "• TF-IDF unigram/bigram features for TF-IDF+LR     • Transformer subword tokenization; BiLSTM per-condition word vocabulary",
    fc="#e6f5ea", fontsize=8.3)
arrow(66, feat_y, 66, feat_y - 2.3)

# Models
mod_y, mod_h = 44.5, 7.5
ax.text(66, mod_y + mod_h + 1.3, "Models (6 architectures)", ha="center", fontsize=9, weight="bold")
model_labels = [
    "TF-IDF +\nLogistic Regression\n(Bag-of-words baseline)",
    "RoBERTa-base\n(125M params)\nBidirectional attention",
    "BERT-base-cased\n(110M params)\nBidirectional attention",
    "ALBERT-base-v2\n(12M params)\nBidirectional attention\nwith parameter sharing",
    "GPT-2\n(124M params)\nCausal (attends to\npast only)",
    "BiLSTM\n(From scratch)\nBidirectional recurrence,\nno attention",
]
mw = 20
mgap = 0.8
mstart = 3
for i, text in enumerate(model_labels):
    x = mstart + i * (mw + mgap)
    box(x, mod_y, mw, mod_h, text, fc="#fdf1de", fontsize=7)
arrow(66, mod_y, 66, mod_y - 2.3)

# Training setup -- CORRECTED CLAIM
train_y, train_h = 34.5, 8
box(3, train_y, 126, train_h,
    "Training setup for the main nine-condition grid\n"
    "• Class-balanced loss     • Same train/test utterance split across all nine conditions within each corpus "
    "(the transcript TEXT differs per condition; the underlying rows do not)\n"
    "• Transformers: 3 epochs, LR = 2e−5, batch size 16, fp16     • BiLSTM: 15 epochs, per-condition vocab, emb dim = 200, hidden dim = 128",
    fc="#efe9f7", fontsize=7.6)
arrow(66, train_y, 66, train_y - 2.3)

# Evaluation
eval_y, eval_h = 26.5, 6.5
box(3, eval_y, 126, eval_h,
    "Evaluation (same protocol for every model × condition × corpus)\n"
    "• Macro-F1   • Micro-F1 (Accuracy)   • McNemar's test (paired utterance-level)   "
    "• Conversation-level permutation test (Switchboard only)   • Per-label F1   • Confusion analysis",
    fc="#fbe9ec", fontsize=7.6)
arrow(66, eval_y, 66, eval_y - 2.3)

# Outcome
out_y, out_h = 19.5, 5.5
box(3, out_y, 126, out_h,
    "Outcome: Directly compare how each normalization operation affects dialogue-act classification\n"
    "across models with different contextual modeling capacities and across conversational registers.\n"
    "Real-ASR results (asr_real) evaluate recognition errors separately.",
    fc="#eaf1fc", fontsize=8, weight="bold")

# Side boxes
box(131, 44.5, 21, 20,
    "Corpora (3 total)\n\n"
    "• Switchboard (Telephone, 27 labels)\n"
    "• MRDA (Meetings, 12 labels)\n"
    "• SLUE-HVB (Call-center, 16 labels)\n\n"
    "Key principles\n\n"
    "• Each condition transforms both\n  train and test text the same way.\n"
    "• Real-ASR results evaluate\n  recognition errors separately.",
    fc="white", fontsize=6.8, ha="left")
arrow(141, 44.5, 129, eval_y + eval_h / 2, color="#2a7a2a")

fig.tight_layout()
fig.savefig("paper/pipeline_diagram_v2.png", dpi=200)
print("Saved pipeline_diagram_v2.png")
