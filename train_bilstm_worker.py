#!/usr/bin/env python3
"""
Trains a from-scratch BiLSTM classifier on exactly ONE normalization condition
and exits, mirroring train_transformer_worker.py's CLI so it can be driven by
the same orchestrator-style retry loop.

This is the genuinely non-Transformer, non-attention architecture requested to
separate two hypotheses about why RoBERTa/BERT/ALBERT recover from filler
removal: (a) it's specifically self-attention letting the model look at the
whole context at once, or (b) it's just "sequential/contextual processing
beats bag-of-words" more generally. A BiLSTM processes tokens sequentially
(no attention) but is still context-sensitive (each position's hidden state
depends on the whole sequence via forward+backward passes), unlike TF-IDF's
pure bag-of-words counts. There is no pretrained checkpoint for this
architecture (unlike BERT/RoBERTa/ALBERT/GPT-2) -- it is trained end-to-end
from scratch on each condition's training split, embeddings included.

Usage: python3 train_bilstm_worker.py <condition> <out_dir> [dataset]
"""
import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_transformer_worker import load_train_test, MIN_COUNT  # noqa: E402

MAX_LEN = 96
EMB_DIM = 200
HIDDEN_DIM = 128
NUM_EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
MIN_WORD_FREQ = 2


def tokenize(text):
    return re.findall(r"\S+", (text or "").lower())


class Vocab:
    def __init__(self, texts, min_freq=MIN_WORD_FREQ):
        from collections import Counter
        counts = Counter()
        for t in texts:
            counts.update(tokenize(t))
        self.word2id = {"<pad>": 0, "<unk>": 1}
        for w, c in counts.items():
            if c >= min_freq:
                self.word2id[w] = len(self.word2id)

    def __len__(self):
        return len(self.word2id)

    def encode(self, text, max_len=MAX_LEN):
        ids = [self.word2id.get(w, 1) for w in tokenize(text)[:max_len]]
        ids = ids + [0] * (max_len - len(ids))
        return ids


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.encoded = [vocab.encode(t) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 4, num_labels)  # concat(max, mean) of bidirectional outputs

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = out * mask
        has_any = (mask.sum(dim=1) > 0)  # (batch, 1): False for all-padding (empty-text) rows
        max_pool = out.masked_fill(mask == 0, -1e9).max(dim=1).values
        max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))
        mean_pool = out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.cat([max_pool, mean_pool], dim=1)
        return self.classifier(self.dropout(pooled))


def main():
    cond = sys.argv[1]
    OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results_bilstm"
    DATASET = sys.argv[3] if len(sys.argv) > 3 else "switchboard"
    RUN_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else SEED
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(RUN_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df, test_df = load_train_test(DATASET)
    labels_order = sorted(train_df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels_order)}

    y_train_ids = train_df["label"].map(label2id).to_numpy()
    y_test_ids = test_df["label"].map(label2id).to_numpy()

    train_texts = train_df[cond].fillna("").tolist()
    test_texts = test_df[cond].fillna("").tolist()

    vocab = Vocab(train_texts)
    print(f"Vocab size: {len(vocab)}", flush=True)

    train_ds = TextDataset(train_texts, y_train_ids, vocab)
    test_ds = TextDataset(test_texts, y_test_ids, vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    counts = np.bincount(y_train_ids, minlength=len(labels_order)).astype(float)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(len(y_train_ids) / (len(labels_order) * counts), dtype=torch.float).to(device)

    model = BiLSTMClassifier(len(vocab), len(labels_order)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"epoch {epoch+1}/{NUM_EPOCHS} loss={total_loss/len(train_ds):.4f}", flush=True)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    pred_ids = np.concatenate(all_preds)

    macro = f1_score(y_test_ids, pred_ids, average="macro", zero_division=0)
    micro = f1_score(y_test_ids, pred_ids, average="micro", zero_division=0)
    acc = accuracy_score(y_test_ids, pred_ids)

    np.save(os.path.join(OUT_DIR, f"preds_{cond}.npy"), pred_ids)
    np.save(os.path.join(OUT_DIR, "y_test_ids.npy"), y_test_ids)
    with open(os.path.join(OUT_DIR, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print(f"RESULT_OK {macro:.6f} {micro:.6f} {acc:.6f}")


if __name__ == "__main__":
    main()
