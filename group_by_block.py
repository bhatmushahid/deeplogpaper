import ast
import json
import pandas as pd
from collections import defaultdict

CSV_INPUT  = "structured_hdfs.csv"
OUTPUT_JSON = "block_sequences.json"
OUTPUT_CSV  = "block_sequences.csv"

print("Loading structured_hdfs.csv...")
df = pd.read_csv(CSV_INPUT)
print(f"  Loaded {len(df):,} lines")

# ── Step 1: Extract block_id from Parameters column ─────────────────
print("\nStep 1: Extracting block_id from parameters...")

def extract_block_id(param_str):
    """Find the token starting with blk_ in the parameters list."""
    try:
        params = ast.literal_eval(param_str)
        for p in params:
            if str(p).startswith("blk_"):
                return p
    except:
        pass
    return None

df["block_id"] = df["Parameters"].apply(extract_block_id)

# Lines with no block_id (some log types may not have one)
no_block = df["block_id"].isna().sum()
print(f"  Lines with block_id    : {df['block_id'].notna().sum():,}")
print(f"  Lines without block_id : {no_block:,}")

# Keep only lines that have a block_id
df_blocks = df[df["block_id"].notna()].copy()

# ── Step 2: Group by block_id, sort by LineId ───────────────────────
print("\nStep 2: Grouping by block_id...")
grouped = (
    df_blocks
    .sort_values("LineId")
    .groupby("block_id")["LogKeyID"]
    .apply(list)
    .to_dict()
)
print(f"  Unique blocks (sessions) found: {len(grouped):,}")

# ── Step 3 & 4: Save as JSON and CSV ────────────────────────────────
print("\nStep 3 & 4: Saving outputs...")

# Save as JSON: {block_id: [key1, key2, ...]}
with open(OUTPUT_JSON, "w") as f:
    json.dump(grouped, f, indent=2)
print(f"  Saved -> {OUTPUT_JSON}")

# Save as CSV for easy viewing
rows = [{"block_id": bid, "sequence": str(seq), "length": len(seq)}
        for bid, seq in grouped.items()]
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"  Saved -> {OUTPUT_CSV}")

# ── Stats ────────────────────────────────────────────────────────────
lengths = [len(v) for v in grouped.values()]
print("\n" + "=" * 55)
print("STATS")
print("=" * 55)
print(f"  Total blocks (sessions) : {len(grouped):,}")
print(f"  Sequence length — min   : {min(lengths)}")
print(f"  Sequence length — max   : {max(lengths)}")
print(f"  Sequence length — avg   : {sum(lengths)/len(lengths):.1f}")

# Preview first 3 blocks
print("\n  Preview (first 3 blocks):")
for bid, seq in list(grouped.items())[:3]:
    print(f"  {bid} -> {seq}")
