#!/usr/bin/env python3
"""
Export utterance-level CSV from ConvoKit's switchboard-processed-corpus.

What this script does:
1) Downloads the corpus into ./data (next to this script)
2) Loads it as a ConvoKit Corpus
3) Extracts utterance text + conversation_id + speaker_id + a dialogue-act label
4) Saves utterances.csv in the same folder as this script
"""

import os
import pandas as pd
from convokit import Corpus, download


def get_label(meta):
    """
    Try multiple common metadata fields to find a dialogue-act label.
    Returns a single string label or None.
    """
    # Most direct single-label keys
    for key in ["damsl_act_tag", "tag", "label"]:
        if key in meta and meta[key] is not None:
            return meta[key]

    # tags may be list like ["sd"] or a string
    if "tags" in meta and meta["tags"] is not None:
        t = meta["tags"]
        if isinstance(t, list) and len(t) > 0:
            return t[0]
        if isinstance(t, str):
            return t

    # full_tags can be list/dict/string depending on corpus version
    if "full_tags" in meta and meta["full_tags"] is not None:
        ft = meta["full_tags"]

        if isinstance(ft, list) and len(ft) > 0:
            # could be list of tags or list of dicts
            if isinstance(ft[0], dict):
                for cand_key in ["tag", "damsl_act_tag", "label"]:
                    if cand_key in ft[0] and ft[0][cand_key] is not None:
                        return ft[0][cand_key]
                return str(ft[0])
            return ft[0]

        if isinstance(ft, dict) and len(ft) > 0:
            first_val = next(iter(ft.values()))
            return first_val if isinstance(first_val, str) else str(first_val)

        if isinstance(ft, str):
            return ft

    return None


def main():
    # Make paths relative to THIS script so everything lands in the same folder.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Download corpus into ./data
    # NOTE: Some older ConvoKit versions may not support data_dir.
    corpus_path = download("switchboard-processed-corpus", data_dir=data_dir)
    corpus = Corpus(filename=corpus_path)

    # Optional sanity check
    corpus.print_summary_stats()

    # Build utterance table
    rows = []
    for utt in corpus.iter_utterances():
        meta = utt.meta
        rows.append({
            "utt_id": utt.id,
            "conversation_id": utt.conversation_id,
            "speaker_id": utt.speaker.id if utt.speaker is not None else None,
            "text": utt.text,
            "label": get_label(meta),

            # Debug / trace fields (safe to remove later)
            "tags_raw": str(meta.get("tags", None)),
            "full_tags_raw": str(meta.get("full_tags", None)),
            "meta_keys": ",".join(sorted(list(meta.keys()))),
        })

    df = pd.DataFrame(rows)

    # Save CSV next to this script
    out_csv = os.path.join(base_dir, "utterances.csv")
    df.to_csv(out_csv, index=False)

    # Prints
    print("\nSaved:", df.shape)
    print("CSV saved to:", out_csv)
    print("Dataset folder:", corpus_path)
    print("Non-null labels:", df["label"].notna().mean())
    print("\nTop 20 labels:\n", df["label"].value_counts().head(20))


if __name__ == "__main__":
    main()
