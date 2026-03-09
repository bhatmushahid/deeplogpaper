import re
import csv
from drain3 import TemplateMiner

LOG_FILE = "data/HDFS_v1/HDFS.log"
CSV_OUTPUT = "parsed_hdfs.csv"

miner = TemplateMiner()
results = []

print("Parsing logs... this may take a few minutes\n")

with open(LOG_FILE) as f:
    for line_id, raw_line in enumerate(f, start=1):
        raw_line = raw_line.rstrip()
        if not raw_line:
            continue

        match = re.match(r'^\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+', raw_line)
        message = raw_line[match.end():] if match else raw_line

        result = miner.add_log_message(message)

        # Print first 20 lines + any time a new template is discovered
        if line_id <= 20 or result["change_type"] == "cluster_created":
            print(f"Line {line_id:>7} | Key #{result['cluster_id']:>3} | {result['template_mined']}")

        results.append({
            "LineId": line_id,
            "LogKey": result["cluster_id"],
            "RawMessage": raw_line
        })

        if line_id % 500000 == 0:
            print(f"  ... processed {line_id:,} lines")

# Save CSV
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["LineId", "LogKey", "RawMessage"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone! Parsed {len(results):,} lines")
print(f"Saved to {CSV_OUTPUT}")
print(f"\nUnique templates found: {len(miner.drain.clusters)}")
for cluster in miner.drain.clusters:
    print(f"  Key #{cluster.cluster_id:>3} ({cluster.size:>8,} lines) -> {cluster.get_template()}")
