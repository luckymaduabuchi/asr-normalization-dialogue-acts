#!/usr/bin/env python3
"""
Generates real ASR transcripts for every retained HVB utterance (train+test)
by running Whisper over the corpus's own audio field (asapp/slue-phase-2,
"hvb" config). Writes results incrementally per split (resumable: skips
orig_idx values already transcribed) so a crash mid-run loses at most one
batch, not the whole pass.

Row alignment to the existing results/hvb/tfidf/utterances_hvb.csv was verified
separately (rebuilding the same filtered dataframe from a fresh dataset load
reproduces it row-for-row, in order, per split) -- orig_idx here is the
position in the underlying HF dataset split, which is what indexes the audio.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

OUT_DIR = "results/hvb/tfidf"
MODEL_NAME = "openai/whisper-base.en"
BATCH_SIZE = 8
MIN_COUNT = 50


def build_filtered_index():
    ds = load_dataset("asapp/slue-phase-2", "hvb")

    def to_df(split_name):
        d = ds[split_name]
        df = pd.DataFrame({
            "issue_id": d["issue_id"],
            "speaker_id": d["speaker_id"],
            "role": d["role"],
            "gold_text": d["text"],
            "label": [labs[0] if len(labs) > 0 else None for labs in d["dialog_acts"]],
        })
        df["split"] = split_name
        df["orig_idx"] = range(len(df))
        return df

    train_df = to_df("train")
    test_df = to_df("test")
    full = pd.concat([train_df, test_df], ignore_index=True)
    full = full.dropna(subset=["label", "gold_text"]).copy()
    full = full[full["gold_text"].str.strip() != ""].copy()
    vc = full["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    full = full[full["label"].isin(keep)].copy()
    return ds, full


def transcribe_split(ds, split_name, orig_indices, processor, model, device):
    out_path = os.path.join(OUT_DIR, f"asr_real_{split_name}.csv")
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = set(prev["orig_idx"].tolist())
        print(f"[{split_name}] resuming, {len(done)} already done", flush=True)

    todo = [i for i in orig_indices if i not in done]
    print(f"[{split_name}] {len(todo)} remaining of {len(orig_indices)}", flush=True)

    d = ds[split_name]
    t0 = time.time()
    for start in range(0, len(todo), BATCH_SIZE):
        batch_idx = todo[start:start + BATCH_SIZE]
        audios = [d[int(i)]["audio"]["array"] for i in batch_idx]
        srs = [d[int(i)]["audio"]["sampling_rate"] for i in batch_idx]
        assert all(sr == 16000 for sr in srs), "expected 16kHz audio"

        inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        attn = inputs.get("attention_mask")

        for attempt in range(1, 8):
            try:
                input_features = inputs.input_features.to(device=device, dtype=model.dtype)
                gen_kwargs = {"max_new_tokens": 128}
                if attn is not None:
                    gen_kwargs["attention_mask"] = attn.to(device)
                with torch.no_grad():
                    pred_ids = model.generate(input_features, **gen_kwargs)
                texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
                texts = [t.strip() for t in texts]
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                wait = min(5 * attempt, 30)
                print(f"[{split_name}] OOM on batch at {start}, attempt {attempt}/7, "
                      f"retrying in {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Repeated OOM at batch start={start}, giving up this run "
                                f"(resumable -- rerun the script to continue).")

        rows = pd.DataFrame({"orig_idx": batch_idx, "asr_real": texts})
        rows.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

        n_done = start + len(batch_idx)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta_min = (len(todo) - n_done) / rate / 60 if rate > 0 else float("nan")
        print(f"[{split_name}] {n_done}/{len(todo)} ({rate:.1f} utt/s, ETA {eta_min:.1f} min)", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading dataset + building filtered index...", flush=True)
    ds, full = build_filtered_index()
    full.to_csv(os.path.join(OUT_DIR, "hvb_filtered_index.csv"), index=False)
    print("Total retained rows:", len(full), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_NAME}...", flush=True)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    for split_name in ["train", "test"]:
        orig_indices = full.loc[full["split"] == split_name, "orig_idx"].tolist()
        transcribe_split(ds, split_name, orig_indices, processor, model, device)

    print("ASR generation complete.", flush=True)


if __name__ == "__main__":
    main()
