"""Elexon half-hourly generation by fuel type (dataset FUELHH).

This is the supply side: how much electricity each fuel type produced in
every half hour (wind, gas "CCGT", nuclear, biomass, interconnector imports
"INT...", and so on).

Data is requested one calendar month at a time:

    Elexon API ──download──> data/raw/elexon_fuelhh/YYYY-MM.json
               ──parse────> one row per (half hour, fuel type), UTC time
               ──validate─> data contract (bad rows go to quarantine)
               ──store────> data/processed/elexon_fuelhh/YYYY-MM.parquet

Stored in "long" format (one row per fuel type) because the set of fuel
types changes over time as new interconnectors open.
"""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from gridcast.config import Settings
from gridcast.contracts import GENERATION_SCHEMA
from gridcast.errors import SourceError
from gridcast.http import get_with_retry
from gridcast.ingest.common import is_recent, validate_and_store
from gridcast.logging import get_logger
from gridcast.settlement import to_utc
from gridcast.storage import write_bytes_atomic

log = get_logger(__name__)

SOURCE = "elexon_fuelhh"
ENDPOINT = "/datasets/FUELHH/stream"


@dataclass(frozen=True)
class MonthReport:
    month: str  # "YYYY-MM"
    rows: int
    quarantined: int
    downloaded: bool


def months_between(start: date, end: date) -> list[tuple[int, int]]:
    """Every (year, month) from start's month to end's month, inclusive."""
    months = pd.period_range(start=start, end=end, freq="M")
    return [(p.year, p.month) for p in months]


def raw_path(settings: Settings, year: int, month: int) -> Path:
    return settings.raw_dir / SOURCE / f"{year:04d}-{month:02d}.json"


def processed_path(settings: Settings, year: int, month: int) -> Path:
    return settings.processed_dir / SOURCE / f"{year:04d}-{month:02d}.parquet"


def quarantine_path(settings: Settings, year: int, month: int) -> Path:
    return settings.data_dir / "quarantine" / SOURCE / f"{year:04d}-{month:02d}.parquet"


def download_month(
    client: httpx.Client, settings: Settings, year: int, month: int, force: bool
) -> tuple[list[dict[str, Any]], bool]:
    """Fetch one month of FUELHH records (or read the cached copy)."""
    path = raw_path(settings, year, month)
    if path.exists() and not force:
        return json.loads(path.read_text()), False

    first = date(year, month, 1)
    last = (pd.Timestamp(first) + pd.offsets.MonthEnd(0)).date()
    response = get_with_retry(
        client,
        f"{settings.elexon_api_base}{ENDPOINT}",
        settings,
        params={"settlementDateFrom": first.isoformat(), "settlementDateTo": last.isoformat()},
    )
    records = response.json()
    if not isinstance(records, list):
        raise SourceError(f"Expected a JSON list from Elexon, got {type(records).__name__}")
    write_bytes_atomic(response.content, path)
    log.info("elexon.download.done", month=f"{year}-{month:02d}", records=len(records))
    return records, True


def parse(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn raw FUELHH records into a clean long table (not yet validated)."""
    df = pd.DataFrame.from_records(records)
    required = {"settlementDate", "settlementPeriod", "fuelType", "generation", "startTime"}
    missing = required - set(df.columns)
    if missing:
        raise SourceError(f"Elexon FUELHH records missing fields: {sorted(missing)}")

    # If Elexon republished a period, keep the latest version.
    if "publishTime" in df.columns:
        df = df.sort_values("publishTime")
    df = df.drop_duplicates(subset=["settlementDate", "settlementPeriod", "fuelType"], keep="last")

    out = pd.DataFrame(
        {
            "settlement_date": pd.to_datetime(df["settlementDate"], format="%Y-%m-%d"),
            "settlement_period": pd.to_numeric(df["settlementPeriod"]).astype("int64"),
            "fuel_type": df["fuelType"].astype(str).str.strip().str.upper(),
            "generation_mw": pd.to_numeric(df["generation"], errors="coerce").astype("float64"),
            "source_start_utc": pd.to_datetime(df["startTime"], utc=True).dt.as_unit("ns"),
        }
    )
    out["timestamp_utc"] = to_utc(out["settlement_date"], out["settlement_period"])
    return out.sort_values(["timestamp_utc", "fuel_type"]).reset_index(drop=True)


def ingest(
    client: httpx.Client,
    settings: Settings,
    start: date,
    end: date,
    force: bool = False,
    today: date | None = None,
) -> list[MonthReport]:
    """Download, clean, validate and store every month from start to end."""
    today_ts = pd.Timestamp(today or datetime.now(UTC).date())
    reports: list[MonthReport] = []

    for year, month in months_between(start, end):
        refresh = force or is_recent(year, month, today_ts)
        records, downloaded = download_month(client, settings, year, month, refresh)
        if not records:
            log.warning("elexon.month_empty", month=f"{year}-{month:02d}")
            continue

        rows, bad = validate_and_store(
            parse(records),
            GENERATION_SCHEMA,
            settings,
            processed_path(settings, year, month),
            quarantine_path(settings, year, month),
        )
        log.info("elexon.month.done", month=f"{year}-{month:02d}", rows=rows, quarantined=bad)
        reports.append(MonthReport(f"{year}-{month:02d}", rows, bad, downloaded))

    return reports
