from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import respx

from gridcast.config import Settings
from gridcast.contracts import GENERATION_SCHEMA, validate_with_quarantine
from gridcast.errors import SourceError
from gridcast.ingest import elexon_generation as eg

STREAM = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH/stream"


def record(
    day: str,
    period: int,
    fuel: str,
    mw: float,
    start: str | None = None,
    published: str = "2024-06-02T00:00:00Z",
) -> dict[str, Any]:
    if start is None:
        start = (
            pd.Timestamp(day).tz_localize("Europe/London").tz_convert("UTC")
            + pd.Timedelta(minutes=30 * (period - 1))
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "dataset": "FUELHH",
        "publishTime": published,
        "startTime": start,
        "settlementDate": day,
        "settlementPeriod": period,
        "fuelType": fuel,
        "generation": mw,
    }


def day_of_records(day: str) -> list[dict[str, Any]]:
    return [
        record(day, sp, fuel, mw)
        for sp in range(1, 49)
        for fuel, mw in (("WIND", 8000.0), ("CCGT", 9000.0), ("INTFR", -1500.0))
    ]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, http_backoff_s=0)


def test_months_between_spans_year_boundary() -> None:
    assert eg.months_between(date(2023, 11, 15), date(2024, 2, 1)) == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


def test_parse_produces_long_table_in_utc() -> None:
    df = eg.parse(day_of_records("2024-06-01"))
    assert len(df) == 48 * 3
    assert set(df["fuel_type"]) == {"WIND", "CCGT", "INTFR"}
    # 1 June is in BST, so period 1 starts at 23:00 UTC the previous day.
    assert df["timestamp_utc"].min() == pd.Timestamp("2024-05-31 23:00", tz="UTC")


def test_parse_keeps_latest_republished_value() -> None:
    records = [
        record("2024-06-01", 1, "WIND", 100.0, published="2024-06-01T01:00:00Z"),
        record("2024-06-01", 1, "WIND", 250.0, published="2024-06-01T05:00:00Z"),
    ]
    df = eg.parse(records)
    assert df["generation_mw"].tolist() == [250.0]


def test_parse_rejects_missing_fields() -> None:
    with pytest.raises(SourceError, match="missing fields"):
        eg.parse([{"settlementDate": "2024-06-01"}])


def test_contract_passes_valid_day() -> None:
    valid, bad = validate_with_quarantine(
        eg.parse(day_of_records("2024-06-01")), GENERATION_SCHEMA, 0.01
    )
    assert len(valid) == 144
    assert bad.empty


def test_contract_quarantines_period_whose_time_disagrees_with_elexon() -> None:
    records = day_of_records("2024-06-01")
    records[0] = record("2024-06-01", 1, "WIND", 8000.0, start="2024-06-01T12:00:00Z")
    records += day_of_records("2024-06-02") * 1
    _, bad = validate_with_quarantine(eg.parse(records), GENERATION_SCHEMA, 0.01)
    assert len(bad) == 1


def test_is_recent_logic() -> None:
    from gridcast.ingest.common import is_recent

    today = pd.Timestamp("2026-09-21")
    assert is_recent(2026, 9, today)
    assert is_recent(2026, 8, today)
    assert not is_recent(2026, 7, today)


@respx.mock
def test_ingest_month_calls_api_with_settlement_dates_and_caches(settings: Settings) -> None:
    route = respx.get(STREAM).respond(json=day_of_records("2024-06-01"))
    with httpx.Client() as client:
        eg.ingest(client, settings, date(2024, 6, 1), date(2024, 6, 30), today=date(2026, 9, 21))
        second = eg.ingest(
            client, settings, date(2024, 6, 1), date(2024, 6, 30), today=date(2026, 9, 21)
        )

    assert route.call_count == 1  # second run used the cached raw file
    params = route.calls[0].request.url.params
    assert params["settlementDateFrom"] == "2024-06-01"
    assert params["settlementDateTo"] == "2024-06-30"
    assert second[0].downloaded is False
    stored = pd.read_parquet(eg.processed_path(settings, 2024, 6))
    assert len(stored) == 144


@respx.mock
def test_ingest_always_refreshes_recent_months(settings: Settings) -> None:
    route = respx.get(STREAM).respond(json=day_of_records("2026-09-01"))
    with httpx.Client() as client:
        for _ in range(2):
            eg.ingest(
                client, settings, date(2026, 9, 1), date(2026, 9, 21), today=date(2026, 9, 21)
            )
    assert route.call_count == 2


@respx.mock
def test_empty_month_is_skipped(settings: Settings) -> None:
    respx.get(STREAM).respond(json=[])
    with httpx.Client() as client:
        reports = eg.ingest(
            client, settings, date(2026, 9, 1), date(2026, 9, 1), today=date(2026, 9, 1)
        )
    assert reports == []
