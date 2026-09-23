from pathlib import Path

import pandas as pd

from gridcast.config import Settings
from gridcast.data import load_demand
from gridcast.ingest import neso_demand
from gridcast.storage import write_parquet_atomic


def demand_rows(start: str, periods: int, nd: float) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="30min", tz="UTC")
    return pd.DataFrame({"timestamp_utc": ts, "nd_mw": nd})


def test_load_demand_extends_history_with_update_and_history_wins(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    history = demand_rows("2026-08-31 22:00", 4, nd=20_000)  # ends 23:30
    update = demand_rows("2026-08-31 23:00", 4, nd=99_999)  # overlaps 2, adds 2

    write_parquet_atomic(history, neso_demand.processed_path(settings, 2026))
    write_parquet_atomic(
        update, settings.processed_dir / neso_demand.UPDATE_SOURCE / "data.parquet"
    )

    df = load_demand(settings)
    assert len(df) == 6
    assert df["timestamp_utc"].is_monotonic_increasing
    assert df["timestamp_utc"].is_unique
    # Overlapping half hours keep the (revised) history values.
    assert (df["nd_mw"].iloc[:4] == 20_000).all()
    assert (df["nd_mw"].iloc[4:] == 99_999).all()


def test_load_demand_works_without_update_file(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    write_parquet_atomic(
        demand_rows("2025-01-01", 4, 20_000), neso_demand.processed_path(settings, 2025)
    )
    assert len(load_demand(settings)) == 4
