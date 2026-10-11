# per_label_drop.py
# Prints top labels with largest per-label F1 drop (gold_text vs rm_fillers_only)
# and writes a CSV you can plot for a figure.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

DATA = "utterances_with_ablations.csv"
RANDOM_STATE = 42
MIN_COUNT = 50
TOPK = 10

df = pd.read_csv(DATA).dropna(subset=["label"]).copy()

# Filter rare labels for stable per-label scores
vc = df["label"].value_counts()
keep = vc[vc >= MIN_COUNT].index
df = df[df["label"].isin(keep)].copy()

# Conversation-level split (same as your baseline)
conv_ids = df["conversation_id"].unique()
train_convs, test_convs = train_test_split(conv_ids, test_size=0.2, random_state=RANDOM_STATE)

train_df = df[df["conversation_id"].isin(train_convs)].copy()
test_df  = df[df["conversation_id"].isin(test_convs)].copy()

labels = sorted(df["label"].unique())

def train_and_predict(text_col):
    X_train = train_df[text_col].fillna("")
    y_train = train_df["label"]
    X_test  = test_df[text_col].fillna("")
    y_test  = test_df["label"]

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95)),
        ("lr", LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return y_test.to_numpy(), pred

def per_label_metrics(y_true, y_pred):
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = pd.DataFrame({
        "label": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": sup
    })
    return out

# Predictions for both conditions (same split)
y_true_g, y_pred_g = train_and_predict("gold_text")
y_true_f, y_pred_f = train_and_predict("rm_fillers_only")

assert np.array_equal(y_true_g, y_true_f), "Mismatch: y_true differs across conditions."
y_true = y_true_g

m_gold = per_label_metrics(y_true, y_pred_g).rename(columns={
    "precision": "precision_gold",
    "recall": "recall_gold",
    "f1": "f1_gold"
})
m_fill = per_label_metrics(y_true, y_pred_f).rename(columns={
    "precision": "precision_fill",
    "recall": "recall_fill",
    "f1": "f1_fill"
})

merged = m_gold.merge(m_fill[["label","precision_fill","recall_fill","f1_fill"]], on="label", how="inner")
merged["f1_drop"] = merged["f1_gold"] - merged["f1_fill"]
merged = merged.sort_values("f1_drop", ascending=False)

print("\n=== Top labels by F1 drop (gold_text - rm_fillers_only) ===")
print(merged[["label","support","f1_gold","f1_fill","f1_drop"]].head(TOPK).to_string(index=False))

# Save full per-label table for plotting
merged.to_csv("per_label_f1_drop.csv", index=False)
print("\nSaved: per_label_f1_drop.csv")

# Optional: show which confusions increased the most (top 10 pairs)
cm_g = confusion_matrix(y_true, y_pred_g, labels=labels)
cm_f = confusion_matrix(y_true, y_pred_f, labels=labels)

# normalize by true label support to make comparable
row_sums = cm_g.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_gn = cm_g / row_sums
cm_fn = cm_f / row_sums

delta = cm_fn - cm_gn  # positive means confusion increased after filler removal

pairs = []
for i, true_lab in enumerate(labels):
    for j, pred_lab in enumerate(labels):
        if i == j:
            continue
        pairs.append((true_lab, pred_lab, delta[i, j]))

pairs.sort(key=lambda x: x[2], reverse=True)

print("\n=== Top increased confusions after filler removal (normalized) ===")
for true_lab, pred_lab, d in pairs[:10]:
    if d <= 0:
        break
    print(f"true={true_lab:>4s}  pred={pred_lab:>4s}  +{d:.4f}")
