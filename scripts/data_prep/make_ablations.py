import pandas as pd
import re

df = pd.read_csv("utterances_text_conditions.csv")

FILLERS = set(["um","uh","erm","mm","hmm","mhm","uhhuh","uh-huh","huh"])
DISCOURSE = set(["like","well"])  # keep simple; expand later

def rm_punct_only(s: str) -> str:
    s = (s or "")
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def rm_fillers_only(s: str) -> str:
    s = (s or "")
    toks = s.split()
    toks = [t for t in toks if t.lower() not in FILLERS]
    return " ".join(toks)

def rm_discourse_only(s: str) -> str:
    s = (s or "")
    toks = s.split()
    toks = [t for t in toks if t.lower() not in DISCOURSE]
    return " ".join(toks)

def norm_colloq_only(s: str) -> str:
    s = (s or "")
    s = re.sub(r"\bgonna\b", "going to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwanna\b", "want to", s, flags=re.IGNORECASE)
    s = re.sub(r"\bkinda\b", "kind of", s, flags=re.IGNORECASE)
    return s

df["rm_punct_only"] = df["gold_text"].astype(str).map(rm_punct_only)
df["rm_fillers_only"] = df["gold_text"].astype(str).map(rm_fillers_only)
df["rm_discourse_only"] = df["gold_text"].astype(str).map(rm_discourse_only)
df["norm_colloq_only"] = df["gold_text"].astype(str).map(norm_colloq_only)

df.to_csv("utterances_with_ablations.csv", index=False)
print("Saved:", df.shape)
print("Columns:", [c for c in df.columns if "only" in c])
