"""Download ETTh1 from the public ETDataset repository and build the toy dataset.

A local alternative to `dvc pull` when the DVC remote is unavailable: a fresh
clone with a new LocalStack starts with an empty bucket, and the production
remote lives in an AWS account there is no need to touch.

The upstream bytes differ from the DVC-tracked version (that one was rewritten
with pandas: same content, different float formatting), so `dvc status` reports
them as modified and train.py records `data_matches_dvc_pointer=false`; that is
expected. To version this copy in a local remote: `dvc add` + `dvc push`.

Usage (from core_ml/):
    poetry run python -m scripts.download_data
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
RAW_PATH = Path("data/raw/ETTh1.csv")
TOY_PATH = Path("data/toy/ETTh1_toy.csv")
TOY_ROWS = 1000


def download(url: str, target: Path, timeout: int = 60) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # A fixed https GitHub URL, not user input.
    with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
        target.write_bytes(response.read())


def write_toy(raw: Path, toy: Path, n_rows: int = TOY_ROWS) -> None:
    """Header + the first `n_rows` hourly rows (chronological order).

    Bytes, not text: on Windows write_text turns every "\\n" into "\\r\\n", and
    the toy file's md5 would depend on the OS that built it.
    """
    toy.parent.mkdir(parents=True, exist_ok=True)
    lines = raw.read_bytes().splitlines(keepends=True)
    toy.write_bytes(b"".join(lines[: n_rows + 1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Sobrescribe archivos existentes.")
    args = parser.parse_args()

    if RAW_PATH.exists() and not args.force:
        print(f"{RAW_PATH} already exists (use --force to replace it).")
    else:
        print(f"Downloading {ETTH1_URL} -> {RAW_PATH}")
        download(ETTH1_URL, RAW_PATH)

    if TOY_PATH.exists() and not args.force:
        print(f"{TOY_PATH} already exists (use --force to replace it).")
    else:
        write_toy(RAW_PATH, TOY_PATH)
        print(f"Generado {TOY_PATH} ({TOY_ROWS} filas).")


if __name__ == "__main__":
    main()
