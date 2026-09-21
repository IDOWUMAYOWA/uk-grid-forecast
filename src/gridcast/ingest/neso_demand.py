"""NESO historic demand ingestion.

Pipeline for each year:

    NESO API ──download──> data/raw/neso_demand/demanddata_YYYY.csv      (raw, untouched)
             ──parse────> clean column names, UTC timestamps, drop forecasts
             ──validate─> data contract (bad rows go to quarantine)
             ──store────> data/processed/neso_demand/year=YYYY/data.parquet

The raw CSV is kept exactly as downloaded so we can always re-process it
if our cleaning logic changes, without downloading again.
"""

import io
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from gridcast.config import Settings
from gridcast.contracts import DEMAND_SCHEMA, validate_with_quarantine
from gridcast.errors import SourceError
from gridcast.http import get_with_retry
from gridcast.logging import get_logger
from gridcast.settlement import to_utc
from gridcast.storage import write_bytes_atomic, write_parquet_atomic

log = get_logger(__name__)

# The "Historic Demand Data" dataset on the NESO Data Portal (CKAN).
DATASET_ID = "8f2fe0af-871c-488d-8bad-960426f24601"
SOURCE = "neso_demand"

# NESO has used different date formats in different years' files,
# e.g. "2025-01-15", "15/01/2025", "15-JAN-2025" and "15-Jan-25".
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%b-%Y", "%d-%b-%y")

# Any parsed date outside this range means we guessed the format wrong.
PLAUSIBLE_YEARS = range(2001, 2101)

REQUIRED_COLUMNS = ("SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND", "TSD")

# Source column -> our column. Units are made explicit in the name.
COLUMN_MAP = {
    "SETTLEMENT_DATE": "settlement_date",
    "SETTLEMENT_PERIOD": "settlement_period",
    "ND": "nd_mw",
    "TSD": "tsd_mw",
    "ENGLAND_WALES_DEMAND": "england_wales_demand_mw",
    "EMBEDDED_WIND_GENERATION": "embedded_wind_generation_mw",
    "EMBEDDED_WIND_CAPACITY": "embedded_wind_capacity_mw",
    "EMBEDDED_SOLAR_GENERATION": "embedded_solar_generation_mw",
    "EMBEDDED_SOLAR_CAPACITY": "embedded_solar_capacity_mw",
    "NON_BM_STOR": "non_bm_stor_mw",
    "PUMP_STORAGE_PUMPING": "pump_storage_pumping_mw",
    "SCOTTISH_TRANSFER": "scottish_transfer_mw",
    "IFA_FLOW": "ifa_flow_mw",
    "IFA2_FLOW": "ifa2_flow_mw",
    "BRITNED_FLOW": "britned_flow_mw",
    "MOYLE_FLOW": "moyle_flow_mw",
    "EAST_WEST_FLOW": "east_west_flow_mw",
    "NEMO_FLOW": "nemo_flow_mw",
    "NSL_FLOW": "nsl_flow_mw",
    "ELECLINK_FLOW": "eleclink_flow_mw",
    "VIKING_FLOW": "viking_flow_mw",
    "GREENLINK_FLOW": "greenlink_flow_mw",
}


@dataclass(frozen=True)
class DemandResource:
    """One yearly CSV file published by NESO."""

    year: int
    url: str
    name: str


@dataclass(frozen=True)
class YearReport:
    year: int
    rows: int
    quarantined: int
    downloaded: bool


# ---------------------------------------------------------------- discovery


def list_resources(client: httpx.Client, settings: Settings) -> dict[int, DemandResource]:
    """Ask the NESO catalogue which yearly files exist. Returns {year: resource}."""
    response = get_with_retry(
        client, f"{settings.neso_api_base}/package_show", settings, params={"id": DATASET_ID}
    )
    payload = response.json()
    if not payload.get("success"):
        raise SourceError(f"NESO package_show failed: {payload.get('error')}")

    resources: dict[int, DemandResource] = {}
    for item in payload["result"]["resources"]:
        url = str(item.get("url", ""))
        match = re.search(r"(\d{4})\.csv$", url, flags=re.IGNORECASE)
        if not match:
            continue
        year = int(match.group(1))
        if year in resources:
            log.warning("neso.duplicate_year", year=year, kept=url)
        resources[year] = DemandResource(year=year, url=url, name=str(item.get("name", "")))
    if not resources:
        raise SourceError("NESO catalogue returned no yearly CSV files")
    return resources


# ----------------------------------------------------------------- download


def raw_path(settings: Settings, year: int) -> Path:
    return settings.raw_dir / SOURCE / f"demanddata_{year}.csv"


def processed_path(settings: Settings, year: int) -> Path:
    return settings.processed_dir / SOURCE / f"year={year}" / "data.parquet"


def quarantine_path(settings: Settings, year: int) -> Path:
    return settings.data_dir / "quarantine" / SOURCE / f"year={year}.parquet"


def download(
    client: httpx.Client, resource: DemandResource, settings: Settings, force: bool
) -> tuple[Path, bool]:
    """Download a yearly CSV unless we already have it. Returns (path, downloaded?)."""
    path = raw_path(settings, resource.year)
    if path.exists() and not force:
        log.info("neso.download.skipped", year=resource.year, path=str(path))
        return path, False
    response = get_with_retry(client, resource.url, settings)
    write_bytes_atomic(response.content, path)
    log.info("neso.download.done", year=resource.year, bytes=len(response.content))
    return path, True


# -------------------------------------------------------------------- parse


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse settlement dates, trying each known NESO format in turn."""
    text = raw.astype(str).str.strip()
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if parsed.notna().all() and parsed.dt.year.isin(PLAUSIBLE_YEARS).all():
            return parsed
    sample = text.head(3).tolist()
    raise SourceError(f"Unrecognised SETTLEMENT_DATE format, sample: {sample}")


def parse(csv_bytes: bytes) -> pd.DataFrame:
    """Turn a raw NESO CSV into a clean, typed table (not yet validated)."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [str(c).strip().upper() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SourceError(f"NESO file is missing required columns: {missing}")

    # Newer files mix actuals ('A') with NESO's own forecasts ('F').
    if "FORECAST_ACTUAL_INDICATOR" in df.columns:
        is_actual = df["FORECAST_ACTUAL_INDICATOR"].astype(str).str.strip().str.upper() == "A"
        log.info("neso.parse.forecast_rows_dropped", rows=int((~is_actual).sum()))
        df = df[is_actual]

    df = df[[c for c in COLUMN_MAP if c in df.columns]].rename(columns=COLUMN_MAP)
    df["settlement_date"] = _parse_dates(df["settlement_date"])
    df["settlement_period"] = pd.to_numeric(df["settlement_period"], errors="coerce")
    df = df.dropna(subset=["settlement_period"])
    df["settlement_period"] = df["settlement_period"].astype("int64")

    value_cols = [c for c in df.columns if c.endswith("_mw")]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    before = len(df)
    df = df.drop_duplicates(subset=["settlement_date", "settlement_period"], keep="last")
    if len(df) < before:
        log.warning("neso.parse.duplicates_dropped", rows=before - len(df))

    df["timestamp_utc"] = to_utc(df["settlement_date"], df["settlement_period"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


# ---------------------------------------------------------------------- run


def ingest(
    client: httpx.Client,
    settings: Settings,
    start_year: int,
    end_year: int,
    force: bool = False,
) -> list[YearReport]:
    """Download, clean, validate and store every year in [start_year, end_year]."""
    available = list_resources(client, settings)
    current_year = datetime.now(UTC).year
    reports: list[YearReport] = []

    for year in range(start_year, end_year + 1):
        resource = available.get(year)
        if resource is None:
            log.warning("neso.year_unavailable", year=year)
            continue

        # The current and previous year keep changing (new days, revisions),
        # so we always refresh them. Older years are downloaded once.
        refresh = force or year >= current_year - 1
        path, downloaded = download(client, resource, settings, force=refresh)

        clean = parse(path.read_bytes())
        valid, quarantined = validate_with_quarantine(
            clean, DEMAND_SCHEMA, settings.max_invalid_row_fraction
        )
        write_parquet_atomic(valid, processed_path(settings, year))
        if len(quarantined):
            write_parquet_atomic(quarantined, quarantine_path(settings, year))

        log.info("neso.year.done", year=year, rows=len(valid), quarantined=len(quarantined))
        reports.append(YearReport(year, len(valid), len(quarantined), downloaded))

    return reports
