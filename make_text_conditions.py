from convokit import Corpus
import pandas as pd
import re
corpus = Corpus(filename="/home/vm-user/Desktop/connl/data/switchboard-processed-corpus")

def get_label(meta):
    # Your labels are in tags/full_tags
    if "tags" in meta and meta["tags"] is not None:
        t = meta["tags"]
        if isinstance(t, str):
            # "o qw" -> take last token as DA label
            parts = t.split()
            return parts[-1] if len(parts) else None
    if "full_tags" in meta and meta["full_tags"] is not None:
        ft = meta["full_tags"]
        if isinstance(ft, str):
            # "o;  qw" -> grab last token
            parts = ft.replace(";", " ").split()
            return parts[-1] if len(parts) else None
    return None

FILLERS = set(["um","uh","erm","mm","hmm","mhm","uhhuh","uh-huh","huh","like","youknow"])

def light_clean(s: str) -> str:
    # mimic common normalization: lowercase, strip punctuation, collapse spaces
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def heavy_clean(s: str) -> str:
    s = light_clean(s)
    toks = [t for t in s.split() if t not in FILLERS]
    s = " ".join(toks)
    # normalize a few common colloquial forms (expand as you want)
    s = s.replace("gonna", "going to").replace("wanna", "want to").replace("kinda", "kind of")
    s = re.sub(r"\s+", " ", s).strip()
    return s

rows = []
for utt in corpus.iter_utterances():
    meta = utt.meta
    text = utt.text or ""
    alpha = meta.get("alpha_text", "")

    rows.append({
        "utt_id": utt.id,
        "conversation_id": utt.conversation_id,
        "speaker_id": utt.speaker.id if utt.speaker else None,
        "label": get_label(meta),

        "gold_text": text,                 # original (with punctuation etc.)
        "gold_alpha_text": alpha,          # already normalized by dataset
        "clean_light": light_clean(text),  # your controlled cleaning
        "clean_heavy": heavy_clean(text),  # your stronger cleaning
    })

df = pd.DataFrame(rows)
df.to_csv("utterances_text_conditions.csv", index=False)

print("Saved:", df.shape)
print("Non-null labels:", df["label"].notna().mean())
print(df["label"].value_counts().head(15))
