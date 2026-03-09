import re
import csv
import pandas as pd
from drain3 import TemplateMiner

LOG_FILE  = "data/HDFS_v1/HDFS.log"
CSV_INPUT = "parsed_hdfs.csv"
CSV_OUTPUT = "structured_hdfs.csv"

# ── Step 1: Rebuild Drain3 vocabulary by replaying the log ──────────
print("Step 1: Rebuilding Drain3 vocabulary...")
miner = TemplateMiner()

header_re = re.compile(r'^\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+')

with open(LOG_FILE) as f:
    for raw_line in f:
        raw_line = raw_line.rstrip()
        if not raw_line:
            continue
        match = header_re.match(raw_line)
        message = raw_line[match.end():] if match else raw_line
        miner.add_log_message(message)

# Build vocabulary: {cluster_id -> template string}
vocabulary = {
    cluster.cluster_id: cluster.get_template()
    for cluster in miner.drain.clusters
}

print(f"  Vocabulary built: {len(vocabulary)} unique log keys")
for key_id, template in sorted(vocabulary.items()):
    print(f"  LogKeyID {key_id:>3} -> {template}")

# ── Step 2 & 3: Extract parameters for each line ───────────────────
print("\nStep 2 & 3: Extracting parameters and saving CSV...")

def extract_parameters(raw_message, template):
    """
    Compare raw message tokens against template tokens.
    Wherever template has <*>, the raw token is a parameter.
    """
    raw_tokens      = raw_message.strip().split()
    template_tokens = template.strip().split()

    # If lengths differ, return full message as single parameter
    if len(raw_tokens) != len(template_tokens):
        return raw_tokens

    params = [
        raw for raw, tmpl in zip(raw_tokens, template_tokens)
        if tmpl == "<*>"
    ]
    return params

df = pd.read_csv(CSV_INPUT)

results = []
for _, row in df.iterrows():
    line_id   = row["LineId"]
    log_key_id = row["LogKey"]
    raw_line  = row["RawMessage"]

    # Strip header to get message only
    match = header_re.match(str(raw_line))
    message = raw_line[match.end():] if match else raw_line

    # Get template for this log key
    template = vocabulary.get(log_key_id, "")

    # Extract variable parts
    parameters = extract_parameters(message, template)

    results.append({
        "LineId":     line_id,
        "LogKeyID":   log_key_id,
        "Parameters": str(parameters)   # saved as a list string e.g. ['blk_123', '/10.0.0.1']
    })

    if line_id <= 5:
        print(f"\n  Line {line_id}")
        print(f"    Template   : {template}")
        print(f"    Parameters : {parameters}")

# Save CSV
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["LineId", "LogKeyID", "Parameters"])
    writer.writeheader()
    writer.writerows(results)

# ── Step 4: Print stats ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STATS")
print("=" * 55)
print(f"  Total log lines     : {len(results):,}")
print(f"  Unique log keys     : {len(vocabulary)}")
print(f"\n  Log key distribution:")
key_counts = df["LogKey"].value_counts().sort_index()
for key_id, count in key_counts.items():
    template = vocabulary.get(key_id, "unknown")
    pct = count / len(df) * 100
    print(f"  Key {key_id:>3} ({count:>9,} lines, {pct:5.1f}%) -> {template[:55]}")

print(f"\nSaved -> {CSV_OUTPUT}")
