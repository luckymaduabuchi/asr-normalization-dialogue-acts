#!/usr/bin/env python3
"""
Extends MRDA and HVB to the full 9-condition Switchboard-style sweep by adding
the 6 conditions currently missing (rm_punct_only, norm_colloq_only,
rm_discourse_only, gold_alpha_text, clean_light, asr_noise), using the exact
same lexicons/logic as Switchboard's make_ablations.py / make_asr_noise_condition.py.

gold_alpha_text on Switchboard comes from ConvoKit's own alpha_text metadata
field, which has no equivalent in the MRDA/HVB releases; here it is
constructed as a stricter alphanumeric-only filter (also strips apostrophes,
unlike rm_punct_only) applied to gold_text, documented as a constructed
analog rather than a native corpus field.
"""
import re
import numpy as np
import pandas as pd

SEED = 42
P_DEL, P_SUB, P_INS = 0.08, 0.08, 0.04

FILLERS = set(["um", "uh", "erm", "mm", "mhm", "hmm", "uhhuh", "uh-huh", "mmm"])
DISCOURSE = set(["like", "well"])


def rm_punct_only(s: str) -> str:
    s = (s or "")
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def gold_alpha_text(s: str) -> str:
    s = (s or "")
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_colloq_only(s: str) -> str:
    s = (s or "")
    s = re.sub(r"\bgonna\b", "going to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwanna\b", "want to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bkinda\b", "kind of", s, flags=re.IGNORECASE)
    return s


def rm_discourse_only(s: str) -> str:
    toks = (s or "").split()
    toks = [t for t in toks if t.lower().strip(".,?!") not in DISCOURSE]
    return " ".join(toks)


def light_clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_light(s: str) -> str:
    return light_clean(s)


def make_asr_noise(df, text_col="gold_text"):
    rng = np.random.default_rng(SEED)
    vocab_counts = {}
    for s in df[text_col].fillna(""):
        for tok in re.findall(r"\S+", s):
            vocab_counts[tok] = vocab_counts.get(tok, 0) + 1
    vocab = np.array(list(vocab_counts.keys()))
    counts = np.array(list(vocab_counts.values()), dtype=float)
    probs = counts / counts.sum()

    def noisify(text):
        toks = re.findall(r"\S+", text or "")
        out = []
        for tok in toks:
            r = rng.random()
            if r < P_DEL:
                continue
            elif r < P_DEL + P_SUB:
                out.append(str(rng.choice(vocab, p=probs)))
            else:
                out.append(tok)
            if rng.random() < P_INS:
                out.append(str(rng.choice(vocab, p=probs)))
        return " ".join(out)

    return df[text_col].fillna("").map(noisify)


def extend(path):
    df = pd.read_csv(path)
    print(f"Extending {path}, {df.shape}")
    df["rm_punct_only"] = df["gold_text"].map(rm_punct_only)
    df["gold_alpha_text"] = df["gold_text"].map(gold_alpha_text)
    df["norm_colloq_only"] = df["gold_text"].map(norm_colloq_only)
    df["rm_discourse_only"] = df["gold_text"].map(rm_discourse_only)
    df["clean_light"] = df["gold_text"].map(clean_light)
    df["asr_noise"] = make_asr_noise(df, "gold_text")
    df.to_csv(path, index=False)
    print(f"Saved {path}, columns now: {list(df.columns)}")


if __name__ == "__main__":
    extend("results_mrda/utterances_mrda.csv")
    extend("results_hvb/utterances_hvb.csv")
