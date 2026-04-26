"""Streaming simulator — feeds the test set through the live API in
micro-batches, so you can watch the dashboard come alive.

The brief, Step 6: "you can simulate streaming with micro-batches to make
the project feel more operational."

Usage:
    # In terminal 1:
    cd milestone4
    uvicorn app:app --reload

    # In terminal 2:
    cd milestone4
    python simulate_stream.py --source ../data/raw/BenignTraffic.csv --batch-size 20 --interval 1.0

    # Or mix several classes:
    python simulate_stream.py --source ../data/raw/DDoS-ICMP_Flood.csv --n-records 500

    # Or stream everything:
    python simulate_stream.py --source ../data/processed/clean_dataset.csv --n-records 5000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


def stream_file(
    source: Path,
    url: str,
    batch_size: int,
    interval: float,
    n_records: int | None,
) -> None:
    print(f"Loading {source} ...")
    df = pd.read_csv(source)
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    if n_records is not None:
        df = df.sample(min(n_records, len(df)), random_state=42).reset_index(drop=True)

    # Replace inf with NaN - the API handles NaN via the saved imputer
    df = df.replace([float("inf"), float("-inf")], None)

    print(f"Streaming {len(df)} records in batches of {batch_size} "
          f"at {interval}s intervals -> {url}")

    total, total_alerts = 0, 0
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        records = [
            {k: v for k, v in rec.items() if pd.notnull(v)}
            for rec in batch.to_dict(orient="records")
        ]

        try:
            r = requests.post(
                f"{url}/predict/batch",
                json={"records": records},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            total += len(data["predictions"])
            total_alerts += data["alert_count"]
            print(
                f"  batch {start // batch_size + 1:>4d}: "
                f"{len(records)} records | "
                f"alerts {data['alert_count']:>3d}/{len(records)} "
                f"({data['alert_rate'] * 100:>5.1f}%)"
            )
        except requests.RequestException as e:
            print(f"  request failed: {e}")

        time.sleep(interval)

    print(f"\nDone. Sent {total} records, got {total_alerts} alerts "
          f"({total_alerts / max(total, 1) * 100:.1f}%).")
    print("Open http://localhost:8000/dashboard to see the monitoring view.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a CSV through the IDS API")
    parser.add_argument("--source", type=Path, required=True,
                        help="CSV to stream (any of the raw/cleaned files)")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running API")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between micro-batches")
    parser.add_argument("--n-records", type=int, default=None,
                        help="Cap the number of records streamed (default: all)")
    args = parser.parse_args()

    stream_file(args.source, args.url, args.batch_size, args.interval, args.n_records)


if __name__ == "__main__":
    main()
