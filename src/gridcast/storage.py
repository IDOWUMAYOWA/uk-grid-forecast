"""Reading and writing local data files.

Writes are atomic: we write to a temporary file and rename it, so a crash
mid-write can never leave a half-written file that looks valid.
"""

from pathlib import Path

import pandas as pd


def write_bytes_atomic(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_parquet_dir(directory: Path) -> pd.DataFrame:
    """Read and concatenate every parquet file under a directory."""
    files = sorted(directory.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {directory}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
