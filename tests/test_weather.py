from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import respx

from gridcast.config import Settings
from gridcast.contracts import WEATHER_SCHEMA, validate_with_quarantine
from gridcast.errors import SourceError
from gridcast.ingest import weather

FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def payload(hours: int = 24, temp: float = 12.5) -> dict[str, Any]:
    times = pd.date_range("2024-01-01", periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M")
    values = {v: [50.0] * hours for v in weather.HOURLY_VARIABLES}
    values["temperature_2m"] = [temp] * hours
    values["apparent_temperature"] = [temp - 2] * hours
    return {"latitude": 51.5, "longitude": -0.1, "hourly": {"time": list(times), **values}}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, http_backoff_s=0)


def test_locations_weights_sum_to_one() -> None:
    assert sum(loc.weight for loc in weather.LOCATIONS) == pytest.approx(1.0)


def test_parse_builds_utc_hourly_table() -> None:
    df = weather.parse(payload(), "london")
    assert len(df) == 24
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert (df["location"] == "london").all()


def test_parse_drops_hours_beyond_the_archive() -> None:
    p = payload(hours=3)
    for v in weather.HOURLY_VARIABLES:
        p["hourly"][v][-1] = None
    assert len(weather.parse(p, "london")) == 2


def test_parse_rejects_missing_variable() -> None:
    p = payload()
    del p["hourly"]["wind_speed_10m"]
    with pytest.raises(SourceError, match="wind_speed_10m"):
        weather.parse(p, "london")


def test_contract_catches_impossible_temperature() -> None:
    df = weather.parse(payload(hours=200), "london")
    df.loc[5, "temperature_2m"] = 80.0  # not in Britain
    valid, bad = validate_with_quarantine(df, WEATHER_SCHEMA, 0.01)
    assert len(bad) == 1
    assert len(valid) == 199


@respx.mock
def test_ingest_requests_forecasts_in_utc_for_every_location(settings: Settings) -> None:
    route = respx.get(FORECAST).respond(json=payload())
    with httpx.Client() as client:
        reports = weather.ingest(client, settings, 2024, 2024, today=date(2026, 9, 21))

    assert route.call_count == len(weather.LOCATIONS)
    params = route.calls[0].request.url.params
    assert params["timezone"] == "UTC"
    assert params["start_date"] == "2024-01-01"
    assert params["end_date"] == "2024-12-31"
    assert params["wind_speed_unit"] == "ms"
    assert {r.location for r in reports} == {loc.name for loc in weather.LOCATIONS}


@respx.mock
def test_current_year_request_stops_at_today(settings: Settings) -> None:
    route = respx.get(FORECAST).respond(json=payload())
    with httpx.Client() as client:
        weather.ingest(client, settings, 2026, 2026, today=date(2026, 9, 21))
    assert route.calls[0].request.url.params["end_date"] == "2026-09-21"


@respx.mock
def test_api_error_message_is_surfaced(settings: Settings) -> None:
    respx.get(FORECAST).respond(400, json={"error": True, "reason": "Invalid variable"})
    with httpx.Client() as client, pytest.raises(SourceError, match="Invalid variable"):
        weather.ingest(client, settings, 2024, 2024, today=date(2026, 9, 21))
