#!/usr/bin/env python3
"""
Cuts per-utterance audio segments from the 3 scoped ICSI meeting-mix WAVs
(Bmr013/Bed002/Bro008, 16kHz mono, downloaded from the official CC-BY-4.0
ICSI corpus release) using the timestamps in results_mrda_real/scoped_utterances.csv,
and runs Whisper-base.en over each segment to produce asr_real text.

Same model/pattern as hvb_generate_real_asr.py. Segments come from a single
mixed-headset track (not per-speaker close-talk mics), so cross-talk bleed
into a given speaker's segment is expected and will be documented as a
limitation of this sub-experiment.

Resumable: skips utterance_ids already present in the output CSV.
"""
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

IN_CSV = "results_mrda_real/scoped_utterances.csv"
AUDIO_DIR = "mrda_icsi_audio"
OUT_CSV = "results_mrda_real/asr_real.csv"
MODEL_NAME = "openai/whisper-base.en"
BATCH_SIZE = 8
SR = 16000
PAD = 0.15  # seconds of padding on each side, clipped to file bounds


def load_meeting_audio(meetings):
    audio = {}
    for meeting in meetings:
        path = os.path.join(AUDIO_DIR, f"{meeting}.wav")
        data, sr = sf.read(path, dtype="float32")
        assert sr == SR
        audio[meeting] = data
    return audio


def cut_segment(audio, meeting, start, end):
    a = audio[meeting]
    s = max(0.0, start - PAD)
    e = min(len(a) / SR, end + PAD)
    i0, i1 = int(s * SR), int(e * SR)
    seg = a[i0:i1]
    if len(seg) < int(0.05 * SR):
        seg = np.pad(seg, (0, int(0.05 * SR) - len(seg)))
    return seg


def main():
    os.makedirs("results_mrda_real", exist_ok=True)
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} scoped utterances", flush=True)

    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(prev["utterance_id"].tolist())
        print(f"Resuming, {len(done)} already done", flush=True)

    todo = df[~df["utterance_id"].isin(done)].reset_index(drop=True)
    print(f"{len(todo)} remaining of {len(df)}", flush=True)
    if len(todo) == 0:
        print("Nothing to do.", flush=True)
        return

    print("Loading meeting audio into memory...", flush=True)
    audio = load_meeting_audio(todo["meeting"].unique().tolist())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_NAME}...", flush=True)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    t0 = time.time()
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[start:start + BATCH_SIZE]
        segs = [cut_segment(audio, r.meeting, r.start, r.end) for r in batch.itertuples()]

        inputs = processor(segs, sampling_rate=SR, return_tensors="pt", padding=True)
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
                print(f"OOM on batch at {start}, attempt {attempt}/7, retrying in {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Repeated OOM at batch start={start}, giving up (rerun to resume).")

        rows = pd.DataFrame({"utterance_id": batch["utterance_id"].tolist(), "asr_real": texts})
        rows.to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)

        n_done = start + len(batch)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta_min = (len(todo) - n_done) / rate / 60 if rate > 0 else float("nan")
        print(f"{n_done}/{len(todo)} ({rate:.1f} utt/s, ETA {eta_min:.1f} min)", flush=True)

    print("ASR generation complete.", flush=True)


if __name__ == "__main__":
    main()
