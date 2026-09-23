"""Open-Meteo historical weather forecasts for GB demand centres.

Why forecasts, not actual weather?  When we predict tomorrow's demand we
only have tomorrow's *forecast* weather. If we trained on actual weather,
the model would learn from better information than it will ever have in
production, and its test scores would be misleadingly good. The Historical
Forecast API gives us the forecasts as they were issued, so training data
matches real-life conditions.

One request per location per year:

    Open-Meteo ──download──> data/raw/open_meteo/<location>/YYYY.json
               ──parse────> hourly rows, UTC time, one row per location
               ──validate─> data contract
               ──store────> data/processed/open_meteo/<location>/YYYY.parquet
"""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from gridcast.config import Settings
from gridcast.contracts import WEATHER_SCHEMA
from gridcast.errors import SourceError
from gridcast.http import get_with_retry
from gridcast.ingest.common import validate_and_store
from gridcast.logging import get_logger
from gridcast.storage import write_bytes_atomic

log = get_logger(__name__)

SOURCE = "open_meteo"


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    weight: float  # rough share of GB electricity demand, used in Phase 2


# Major population centres across GB. Weights are approximate regional
# population shares and only need to be roughly right.
LOCATIONS = (
    Location("london", 51.51, -0.13, 0.30),
    Location("birmingham", 52.49, -1.89, 0.17),
    Location("manchester", 53.48, -2.24, 0.17),
    Location("leeds", 53.80, -1.55, 0.12),
    Location("bristol", 51.45, -2.59, 0.10),
    Location("glasgow", 55.86, -4.25, 0.09),
    Location("cardiff", 51.48, -3.18, 0.05),
)

HOURLY_VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
)


@dataclass(frozen=True)
class WeatherReport:
    location: str
    year: int
    rows: int
    quarantined: int
    downloaded: bool


def raw_path(settings: Settings, location: str, year: int) -> Path:
    return settings.raw_dir / SOURCE / location / f"{year}.json"


def processed_path(settings: Settings, location: str, year: int) -> Path:
    return settings.processed_dir / SOURCE / location / f"{year}.parquet"


def quarantine_path(settings: Settings, location: str, year: int) -> Path:
    return settings.data_dir / "quarantine" / SOURCE / location / f"{year}.parquet"


def download_year(
    client: httpx.Client,
    settings: Settings,
    location: Location,
    year: int,
    today: date,
    force: bool,
) -> tuple[dict[str, Any], bool]:
    path = raw_path(settings, location.name, year)
    if path.exists() and not force:
        return json.loads(path.read_text()), False

    end = min(date(year, 12, 31), today)
    response = get_with_retry(
        client,
        f"{settings.open_meteo_forecast_base}/forecast",
        settings,
        params={
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": f"{year}-01-01",
            "end_date": end.isoformat(),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        },
    )
    payload = response.json()
    if payload.get("error"):
        raise SourceError(f"Open-Meteo error: {payload.get('reason')}")
    write_bytes_atomic(response.content, path)
    log.info("weather.download.done", location=location.name, year=year)
    return payload, True


def parse(payload: dict[str, Any], location: str) -> pd.DataFrame:
    """Turn an Open-Meteo response into a clean hourly table."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise SourceError("Open-Meteo response has no hourly data")
    missing = [v for v in HOURLY_VARIABLES if v not in hourly]
    if missing:
        raise SourceError(f"Open-Meteo response missing variables: {missing}")

    df = pd.DataFrame(
        {v: pd.to_numeric(pd.Series(hourly[v]), errors="coerce") for v in HOURLY_VARIABLES}
    )
    df = df.astype("float64")
    # We requested timezone=UTC, so times are UTC but have no offset marker.
    df.insert(
        0,
        "timestamp_utc",
        pd.to_datetime(pd.Series(hourly["time"])).dt.tz_localize("UTC").dt.as_unit("ns"),
    )
    df.insert(1, "location", location)

    # Hours with no values at all are beyond the end of the archive.
    df = df.dropna(subset=list(HOURLY_VARIABLES), how="all")
    return df.reset_index(drop=True)


def ingest(
    client: httpx.Client,
    settings: Settings,
    start_year: int,
    end_year: int,
    force: bool = False,
    today: date | None = None,
) -> list[WeatherReport]:
    today = today or datetime.now(UTC).date()
    reports: list[WeatherReport] = []

    for location in LOCATIONS:
        for year in range(start_year, end_year + 1):
            refresh = force or year >= today.year - (1 if today.month == 1 else 0)
            payload, downloaded = download_year(client, settings, location, year, today, refresh)
            rows, bad = validate_and_store(
                parse(payload, location.name),
                WEATHER_SCHEMA,
                settings,
                processed_path(settings, location.name, year),
                quarantine_path(settings, location.name, year),
            )
            reports.append(WeatherReport(location.name, year, rows, bad, downloaded))

    return reports
