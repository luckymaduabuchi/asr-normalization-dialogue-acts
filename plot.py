# plot_f1_drop.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_label_f1_drop.csv")

# filter low support for a clean figure
df = df[df["support"] >= 50].copy()

top = df.sort_values("f1_drop", ascending=False).head(10)

plt.figure()
plt.bar(top["label"], top["f1_drop"])
plt.xlabel("Dialogue act label")
plt.ylabel("F1 drop (gold_text - rm_fillers_only)")
plt.title("Largest per-label degradation after filler removal")
plt.tight_layout()
plt.savefig("f1_drop_top10.png", dpi=200)
print("Saved: f1_drop_top10.png")