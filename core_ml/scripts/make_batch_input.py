"""Build the batch-scoring input CSV from a raw ETT CSV.

Takes the last `--rows` hourly readings, derives month/day/hour (as
data_processing.py does) and writes the FEATURE_COLUMNS src.batch_inference
expects to `--output`. Used by `make batch-local`, which uploads the result to
the LocalStack bucket.

Usage (from core_ml/):
    poetry run python -m scripts.make_batch_input --rows 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_contracts import FEATURE_COLUMNS


def build_batch_input(raw_csv: str, rows: int) -> pd.DataFrame:
    df = pd.read_csv(raw_csv).tail(rows)
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df["month"], df["day"], df["hour"] = dates.dt.month, dates.dt.day, dates.dt.hour
    return df[list(FEATURE_COLUMNS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_csv", default="data/toy/ETTh1_toy.csv")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--output", default="artifacts_batch/input_data.csv")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_batch_input(args.raw_csv, args.rows).to_csv(output, index=False, lineterminator="\n")
    print(f"Batch input written to {output}")


if __name__ == "__main__":
    main()
