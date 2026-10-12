#!/usr/bin/env python3
"""
Worker for the scoped MRDA real-ASR sub-experiment: trains ONE model on the
existing, large wylupek-derived MRDA train split (gold_text condition -- the
same training data/condition used for that model's row in the main 9x6x3
grid) and evaluates it TWICE on the new real-audio-derived test set (5 ICSI
meetings, official CC-BY-4.0 release): once on the ICSI gold transcript text,
once on the Whisper-base.en asr_real transcript of the same audio spans.
Training data is unchanged from the main grid; only the test-time text
differs between the two eval passes, isolating the effect of genuine ASR
recognition error at test time for a model trained the normal way.

Not the same design as HVB's asr_real (which trains AND tests on Whisper
text, since HVB's full corpus has audio). Here only 5 of 75 ICSI meetings
have obtainable audio, too little to train 6 models on -- so training reuses
the full existing MRDA train split and only the test text is swapped.

Usage: python3 scripts/real_asr/mrda_real_asr_worker.py <model_key> [out_dir]
model_key in {tfidf, roberta-base, bert-base-cased, albert-base-v2, gpt2, bilstm}
Writes <out_dir>/preds_<model_key>_{gold,asr}.npy, y_test_ids.npy, label2id.json.
Prints "RESULT_OK <gold_macro> <gold_micro> <gold_acc> <asr_macro> <asr_micro> <asr_acc>".
"""
import os
import re
import sys
import json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

MIN_COUNT = 50
MAX_LENGTH = 48  # 99th pct of MRDA train gold_text is 33 words; 96 was mostly wasted padding
SEED = 42
TRAIN_CSV = "results/mrda/tfidf/utterances_mrda.csv"
SCOPED_CSV = "results/legacy/mrda_real/scoped_utterances.csv"
ASR_CSV = "results/legacy/mrda_real/asr_real.csv"

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=3, max_df=0.95)
LR_KWARGS = dict(max_iter=2000, n_jobs=-1, class_weight="balanced")


def load_train():
    df = pd.read_csv(TRAIN_CSV).dropna(subset=["label"]).copy()
    vc = df["label"].value_counts()
    keep = vc[vc >= MIN_COUNT].index
    df = df[df["label"].isin(keep)].copy()
    return df[df["split"] == "train"].copy()


def load_test():
    scoped = pd.read_csv(SCOPED_CSV)
    asr = pd.read_csv(ASR_CSV)
    df = scoped.merge(asr, on="utterance_id", how="inner")  # inner drops stale ids from an earlier 3-meeting run
    df = df.rename(columns={"general_da": "label"})
    df["asr_real"] = df["asr_real"].fillna("")
    return df


def metrics(y_true, y_pred, labels_order):
    macro = f1_score(y_true, y_pred, average="macro", labels=list(range(len(labels_order))), zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", labels=list(range(len(labels_order))), zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return macro, micro, acc


def run_tfidf(train_df, test_df, label2id, out_dir):
    y_train = train_df["label"].map(label2id).to_numpy()
    y_test = test_df["label"].map(label2id).to_numpy()
    clf = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_KWARGS)), ("lr", LogisticRegression(**LR_KWARGS))])
    clf.fit(train_df["gold_text"].fillna(""), y_train)
    pred_gold = clf.predict(test_df["gold_text"].fillna(""))
    pred_asr = clf.predict(test_df["asr_real"].fillna(""))
    return y_test, pred_gold, pred_asr


class WeightedTrainer:
    """Manual training loop (avoids transformers.Trainer class-weight plumbing
    differences across model types); mirrors scripts/train/train_transformer_worker.py's
    hyperparameters (3 epochs, batch 16, lr 2e-5, class-weighted CE loss,
    fp16 mixed precision -- this GPU is shared and memory is tight)."""

    def __init__(self, model, class_weights, device):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.device = device
        self.use_amp = device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def fit(self, loader, epochs=3, lr=2e-5):
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            for batch in loader:
                opt.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    out = self.model(input_ids=input_ids, attention_mask=attn)
                    loss = self.loss_fn(out.logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
                total += loss.item() * input_ids.size(0)
            print(f"epoch {epoch+1}/{epochs} loss={total/len(loader.dataset):.4f}", flush=True)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        preds = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(input_ids=input_ids, attention_mask=attn)
            preds.append(out.logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)


def run_transformer(model_name, train_df, test_df, label2id, device):
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    y_train = train_df["label"].map(label2id).to_numpy()
    y_test = test_df["label"].map(label2id).to_numpy()
    labels_order = list(label2id.keys())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels_order), label2id=label2id, id2label={i: l for l, i in label2id.items()}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    def make_loader(texts, labels, shuffle):
        enc = tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
        ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long))

        def collate(batch):
            ids, attn, labs = zip(*batch)
            return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(attn), "labels": torch.stack(labs)}

        return torch.utils.data.DataLoader(ds, batch_size=8 if shuffle else 32, shuffle=shuffle, collate_fn=collate)

    train_loader = make_loader(train_df["gold_text"].fillna(""), y_train, shuffle=True)
    gold_loader = make_loader(test_df["gold_text"].fillna(""), y_test, shuffle=False)
    asr_loader = make_loader(test_df["asr_real"].fillna(""), y_test, shuffle=False)

    counts = np.bincount(y_train, minlength=len(labels_order)).astype(float)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(len(y_train) / (len(labels_order) * counts), dtype=torch.float)

    torch.manual_seed(SEED)
    trainer = WeightedTrainer(model, class_weights, device)
    trainer.fit(train_loader)
    pred_gold = trainer.predict(gold_loader)
    pred_asr = trainer.predict(asr_loader)
    return y_test, pred_gold, pred_asr


def run_bilstm(train_df, test_df, label2id, device):
    from collections import Counter

    def tokenize(text):
        return re.findall(r"\S+", (text or "").lower())

    class Vocab:
        def __init__(self, texts, min_freq=2):
            counts = Counter()
            for t in texts:
                counts.update(tokenize(t))
            self.word2id = {"<pad>": 0, "<unk>": 1}
            for w, c in counts.items():
                if c >= min_freq:
                    self.word2id[w] = len(self.word2id)

        def encode(self, text, max_len=MAX_LENGTH):
            ids = [self.word2id.get(w, 1) for w in tokenize(text)[:max_len]]
            return ids + [0] * (max_len - len(ids))

        def __len__(self):
            return len(self.word2id)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, num_labels, emb_dim=200, hidden_dim=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(hidden_dim * 4, num_labels)

        def forward(self, x):
            mask = (x != 0).unsqueeze(-1).float()
            emb = self.embedding(x)
            out, _ = self.lstm(emb)
            out = out * mask
            has_any = (mask.sum(dim=1) > 0)  # (batch, 1): False for all-padding (empty-text) rows
            max_pool = out.masked_fill(mask == 0, -1e9).max(dim=1).values
            max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))
            mean_pool = out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.classifier(self.dropout(torch.cat([max_pool, mean_pool], dim=1)))

    y_train = train_df["label"].map(label2id).to_numpy()
    y_test = test_df["label"].map(label2id).to_numpy()
    labels_order = list(label2id.keys())

    train_texts = train_df["gold_text"].fillna("").tolist()
    vocab = Vocab(train_texts)
    print(f"Vocab size: {len(vocab)}", flush=True)

    def make_loader(texts, labels, shuffle, batch_size):
        enc = torch.tensor([vocab.encode(t) for t in texts], dtype=torch.long)
        ds = torch.utils.data.TensorDataset(enc, torch.tensor(labels, dtype=torch.long))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_texts, y_train, True, 32)
    gold_loader = make_loader(test_df["gold_text"].fillna(""), y_test, False, 64)
    asr_loader = make_loader(test_df["asr_real"].fillna(""), y_test, False, 64)

    counts = np.bincount(y_train, minlength=len(labels_order)).astype(float)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(len(y_train) / (len(labels_order) * counts), dtype=torch.float).to(device)

    torch.manual_seed(SEED)
    model = BiLSTMClassifier(len(vocab), len(labels_order)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(15):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        print(f"epoch {epoch+1}/15 loss={total/len(train_loader.dataset):.4f}", flush=True)

    def predict(loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                preds.append(model(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    pred_gold = predict(gold_loader)
    pred_asr = predict(asr_loader)
    return y_test, pred_gold, pred_asr


def main():
    model_key = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/legacy/mrda_real"
    os.makedirs(out_dir, exist_ok=True)

    train_df = load_train()
    test_df = load_test()
    labels_order = sorted(train_df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels_order)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_key}] train={len(train_df)} test={len(test_df)} labels={labels_order} device={device}", flush=True)

    if model_key == "tfidf":
        y_test, pred_gold, pred_asr = run_tfidf(train_df, test_df, label2id, out_dir)
    elif model_key == "bilstm":
        y_test, pred_gold, pred_asr = run_bilstm(train_df, test_df, label2id, device)
    else:
        y_test, pred_gold, pred_asr = run_transformer(model_key, train_df, test_df, label2id, device)

    gold_macro, gold_micro, gold_acc = metrics(y_test, pred_gold, labels_order)
    asr_macro, asr_micro, asr_acc = metrics(y_test, pred_asr, labels_order)

    np.save(os.path.join(out_dir, f"preds_{model_key}_gold.npy"), pred_gold)
    np.save(os.path.join(out_dir, f"preds_{model_key}_asr.npy"), pred_asr)
    np.save(os.path.join(out_dir, "y_test_ids.npy"), y_test)
    with open(os.path.join(out_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print(f"RESULT_OK {gold_macro:.6f} {gold_micro:.6f} {gold_acc:.6f} {asr_macro:.6f} {asr_micro:.6f} {asr_acc:.6f}", flush=True)


if __name__ == "__main__":
    main()
