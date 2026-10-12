#!/usr/bin/env python3
"""
Adds an `asr_noise` column to utterances_with_ablations.csv: a synthetic proxy for
real ASR output, built by injecting word-level deletions/substitutions/insertions
into gold_text at rates chosen to land in the ~15-20% WER range commonly reported
for conversational telephone speech (Switchboard-style) ASR systems.

This is NOT a phonetic-confusion simulation (we have no acoustic model or
confusion matrix available in this environment) -- substitutions draw a
frequency-weighted random word from the corpus vocabulary. It is documented in
the paper as a generic WER-matched proxy, not a claim of phonetic realism.

Deterministic given SEED, operates per-utterance so it composes cleanly with the
existing ablation columns.
"""

import re
import numpy as np
import pandas as pd

DATA_IN = "utterances_with_ablations.csv"
DATA_OUT = "utterances_with_ablations.csv"  # write back with new column appended

SEED = 42
P_DEL = 0.08
P_SUB = 0.08
P_INS = 0.04


def tokenize(s: str):
    return re.findall(r"\S+", s or "")


def main():
    df = pd.read_csv(DATA_IN)
    rng = np.random.default_rng(SEED)

    # Build a frequency-weighted vocabulary from gold_text for substitution/insertion draws
    vocab_counts = {}
    for s in df["gold_text"].fillna(""):
        for tok in tokenize(s):
            vocab_counts[tok] = vocab_counts.get(tok, 0) + 1
    vocab = np.array(list(vocab_counts.keys()))
    counts = np.array(list(vocab_counts.values()), dtype=float)
    probs = counts / counts.sum()

    n_del = n_sub = n_ins = n_orig_tokens = 0

    def noisify(text: str) -> str:
        nonlocal n_del, n_sub, n_ins, n_orig_tokens
        toks = tokenize(text)
        n_orig_tokens += len(toks)
        out = []
        for tok in toks:
            r = rng.random()
            if r < P_DEL:
                n_del += 1
                continue  # deletion
            elif r < P_DEL + P_SUB:
                n_sub += 1
                out.append(str(rng.choice(vocab, p=probs)))
            else:
                out.append(tok)
            if rng.random() < P_INS:
                n_ins += 1
                out.append(str(rng.choice(vocab, p=probs)))
        return " ".join(out)

    df["asr_noise"] = df["gold_text"].fillna("").map(noisify)

    total_ops = n_del + n_sub + n_ins
    composite_wer = total_ops / max(n_orig_tokens, 1)
    print(f"Original tokens: {n_orig_tokens}")
    print(f"Deletions: {n_del} ({n_del/n_orig_tokens:.3%})")
    print(f"Substitutions: {n_sub} ({n_sub/n_orig_tokens:.3%})")
    print(f"Insertions: {n_ins} ({n_ins/n_orig_tokens:.3%})")
    print(f"Composite WER proxy (S+D+I)/N: {composite_wer:.3%}")

    df.to_csv(DATA_OUT, index=False)
    print("Saved:", DATA_OUT, df.shape)


if __name__ == "__main__":
    main()
