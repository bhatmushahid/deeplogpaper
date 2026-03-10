import json
import numpy as np
import pandas as pd
from collections import Counter

LABEL_FILE      = "data/HDFS_v1/preprocessed/anomaly_label.csv"
SEQUENCES_FILE  = "block_sequences.json"
OUTPUT_FILE     = "deeplog_windows.npz"
WINDOW_SIZE     = 10

# ── Step 1: Load labels and split normal vs anomalous ───────────────
print("Step 1: Loading labels...")
labels_df = pd.read_csv(LABEL_FILE)

normal_blocks   = set(labels_df[labels_df["Label"] == "Normal"]["BlockId"].tolist())
anomaly_blocks  = set(labels_df[labels_df["Label"] == "Anomaly"]["BlockId"].tolist())

print(f"  Normal blocks    : {len(normal_blocks):,}")
print(f"  Anomaly blocks   : {len(anomaly_blocks):,}")

# ── Step 2: Load block sequences ────────────────────────────────────
print("\nStep 2: Loading block sequences...")
with open(SEQUENCES_FILE) as f:
    sequences = json.load(f)
print(f"  Total sequences loaded: {len(sequences):,}")

# Match with labels
normal_seqs  = {bid: seq for bid, seq in sequences.items() if bid in normal_blocks}
anomaly_seqs = {bid: seq for bid, seq in sequences.items() if bid in anomaly_blocks}
unmatched    = {bid: seq for bid, seq in sequences.items()
                if bid not in normal_blocks and bid not in anomaly_blocks}

print(f"  Matched normal   : {len(normal_seqs):,}")
print(f"  Matched anomaly  : {len(anomaly_seqs):,}")
print(f"  Unmatched        : {len(unmatched):,} (no label — excluded)")

# ── Step 3: Sliding window on normal sequences only ─────────────────
print(f"\nStep 3: Applying sliding window (h={WINDOW_SIZE}) on normal sequences...")

X = []  # inputs  — shape (N, WINDOW_SIZE)
y = []  # labels  — shape (N,)

skipped_short = 0

for bid, seq in normal_seqs.items():
    # Need at least WINDOW_SIZE + 1 events to make one window
    if len(seq) <= WINDOW_SIZE:
        skipped_short += 1
        continue

    for i in range(len(seq) - WINDOW_SIZE):
        window = seq[i : i + WINDOW_SIZE]
        target = seq[i + WINDOW_SIZE]
        X.append(window)
        y.append(target)

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

print(f"  Sequences skipped (too short): {skipped_short:,}")
print(f"  Total windows generated      : {len(X):,}")

# ── Step 4: Save to NumPy file ───────────────────────────────────────
print(f"\nStep 4: Saving to {OUTPUT_FILE}...")
np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    y=y
)
print(f"  Saved -> {OUTPUT_FILE}")

# Verify saved file
loaded = np.load(OUTPUT_FILE)
print(f"  Verified X shape : {loaded['X'].shape}")
print(f"  Verified y shape : {loaded['y'].shape}")

# ── Stats ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STATS")
print("=" * 55)
print(f"  Window size (h)          : {WINDOW_SIZE}")
print(f"  Normal sequences used    : {len(normal_seqs) - skipped_short:,}")
print(f"  Total training windows   : {len(X):,}")
print(f"  Input shape              : {X.shape}  (windows x h)")
print(f"  Label shape              : {y.shape}")
print(f"  Unique log keys in X     : {len(set(X.flatten()))}")
print(f"  Unique log keys in y     : {len(set(y))}")

print(f"\n  Label (next event) distribution — top 10:")
top_labels = Counter(y.tolist()).most_common(10)
for key_id, count in top_labels:
    pct = count / len(y) * 100
    print(f"    LogKey {key_id:>3} : {count:>9,} ({pct:5.1f}%)")

print(f"\n  Sample windows (first 3):")
for i in range(3):
    print(f"    Input: {X[i].tolist()}  ->  Target: {y[i]}")
