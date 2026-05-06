import pandas as pd
import json, os
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

os.makedirs("iclr2026/data", exist_ok=True)

wandb_file = "wandb/run-20260430_203505-775sskze/run-775sskze.wandb"

ds = datastore.DataStore()
ds.open_for_scan(wandb_file)

rows = []
count = 0
while True:
    try:
        data = ds.scan_data()
        if data is None:
            break
        pb = wandb_internal_pb2.Record()
        pb.ParseFromString(data)
        record_type = pb.WhichOneof("record_type")

        # Debug first 3 history records
        if record_type == "history" and count < 3:
            print(f"\n--- History record {count} ---")
            print(f"  Items count: {len(pb.history.item)}")
            for i, item in enumerate(pb.history.item[:5]):
                print(f"  item[{i}]: key='{item.key}' nested_key={item.nested_key} value_json='{item.value_json[:50]}'")
            count += 1

        if record_type == "history":
            row = {}
            for item in pb.history.item:
                # Try both key and nested_key
                key = item.key if item.key else ".".join(item.nested_key)
                if key:
                    try:
                        row[key] = json.loads(item.value_json)
                    except:
                        row[key] = item.value_json
            if row:
                rows.append(row)
    except Exception as e:
        continue

print(f"\nTotal rows: {len(rows)}")
if rows:
    print(f"Sample keys from row 0: {list(rows[0].keys())[:10]}")
    print(f"Sample keys from row 1: {list(rows[1].keys())[:10]}")

    df = pd.DataFrame(rows)
    print(f"DataFrame columns: {list(df.columns)[:15]}")
    df.to_csv("iclr2026/data/muon_mugreats_full.csv", index=False)
    print(f"Saved CSV with {len(df)} rows")
