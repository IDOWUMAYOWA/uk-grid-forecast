import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from gridcast.config import Settings
from gridcast.contracts import DEMAND_SCHEMA, validate_with_quarantine
from gridcast.errors import DataQualityError, SourceError
from gridcast.ingest import neso_demand

FIXTURES = Path(__file__).parent / "fixtures"
API = "https://api.neso.energy/api/3/action"


def make_csv(dates: list[str], nd: list[int] | None = None, extra: str = "") -> bytes:
    """Build a small NESO-style CSV: 48 periods per date."""
    header = "SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,TSD,EMBEDDED_SOLAR_GENERATION" + extra
    lines = [header]
    i = 0
    for d in dates:
        for sp in range(1, 49):
            value = nd[i] if nd and i < len(nd) else 25_000
            lines.append(f"{d},{sp},{value},{value + 2000},0")
            i += 1
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, http_backoff_s=0, http_max_retries=3)


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "date_text", ["2025-01-15", "15/01/2025", "15/01/25", "15-JAN-2025", "15-Jan-25"]
)
def test_parse_handles_every_known_date_format(date_text: str) -> None:
    df = neso_demand.parse(make_csv([date_text]))
    assert (df["settlement_date"] == pd.Timestamp("2025-01-15")).all()
    assert len(df) == 48


def test_parse_renames_columns_and_adds_utc_timestamp() -> None:
    df = neso_demand.parse(make_csv(["2025-01-15"]))
    assert {"nd_mw", "tsd_mw", "embedded_solar_generation_mw", "timestamp_utc"} <= set(df.columns)
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2025-01-15 00:00", tz="UTC")


def test_parse_drops_forecast_rows() -> None:
    csv = (
        b"SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,TSD,FORECAST_ACTUAL_INDICATOR\n"
        b"2026-09-20,1,20000,22000,A\n"
        b"2026-09-20,2,20100,22100,A\n"
        b"2026-09-22,1,21000,23000,F\n"
    )
    df = neso_demand.parse(csv)
    assert len(df) == 2
    assert "forecast_actual_indicator" not in df.columns


def test_parse_drops_duplicate_periods_keeping_latest() -> None:
    csv = (
        b"SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,TSD\n"
        b"2025-01-15,1,20000,22000\n"
        b"2025-01-15,1,20500,22500\n"
    )
    df = neso_demand.parse(csv)
    assert len(df) == 1
    assert df["nd_mw"].iloc[0] == 20500


def test_parse_rejects_missing_required_column() -> None:
    with pytest.raises(SourceError, match="missing required columns"):
        neso_demand.parse(b"SETTLEMENT_DATE,SETTLEMENT_PERIOD,TSD\n2025-01-15,1,22000\n")


def test_parse_rejects_unknown_date_format() -> None:
    with pytest.raises(SourceError, match="Unrecognised"):
        neso_demand.parse(make_csv(["Jan 15th 2025"]))


# --------------------------------------------------------------- contracts


def test_valid_data_passes_contract() -> None:
    df = neso_demand.parse(make_csv(["2025-01-15", "2025-01-16"]))
    valid, quarantined = validate_with_quarantine(df, DEMAND_SCHEMA, max_invalid_fraction=0.01)
    assert len(valid) == 96
    assert quarantined.empty


def test_a_few_bad_rows_are_quarantined_not_fatal() -> None:
    nd = [25_000] * 96 * 3
    nd[10] = 0  # impossible: GB demand is never zero
    df = neso_demand.parse(make_csv(["2025-01-15", "2025-01-16", "2025-01-17"], nd=nd))
    valid, quarantined = validate_with_quarantine(df, DEMAND_SCHEMA, max_invalid_fraction=0.01)
    assert len(quarantined) == 1
    assert quarantined["nd_mw"].iloc[0] == 0
    assert len(valid) == 143


def test_too_many_bad_rows_stops_the_pipeline() -> None:
    nd = [0] * 48
    df = neso_demand.parse(make_csv(["2025-01-15"], nd=nd))
    with pytest.raises(DataQualityError, match="failed validation"):
        validate_with_quarantine(df, DEMAND_SCHEMA, max_invalid_fraction=0.01)


def test_period_49_on_a_normal_day_is_caught() -> None:
    csv = make_csv(["2025-01-15"] * 1) + b"2025-01-15,49,25000,27000,0\n"
    df = neso_demand.parse(csv)
    _, quarantined = validate_with_quarantine(df, DEMAND_SCHEMA, max_invalid_fraction=0.05)
    assert quarantined["settlement_period"].tolist() == [49]


# ------------------------------------------------------ API + end to end


@respx.mock
def test_list_resources_finds_yearly_csvs(settings: Settings) -> None:
    respx.get(f"{API}/package_show").respond(
        json=json.loads((FIXTURES / "neso_package_show.json").read_text())
    )
    with httpx.Client() as client:
        resources = neso_demand.list_resources(client, settings)
    assert sorted(resources) == [2024, 2025, 2026]
    assert resources[2026].url.endswith("demanddataupdate_2026.csv")


@respx.mock
def test_transient_server_error_is_retried(settings: Settings) -> None:
    route = respx.get(f"{API}/package_show")
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(200, json=json.loads((FIXTURES / "neso_package_show.json").read_text())),
    ]
    with httpx.Client() as client:
        resources = neso_demand.list_resources(client, settings)
    assert route.call_count == 2
    assert 2024 in resources


@respx.mock
def test_not_found_is_not_retried(settings: Settings) -> None:
    route = respx.get(f"{API}/package_show").respond(404, text="Not found")
    with httpx.Client() as client, pytest.raises(SourceError, match="404"):
        neso_demand.list_resources(client, settings)
    assert route.call_count == 1


@respx.mock
def test_server_still_down_after_retries_raises_source_error(settings: Settings) -> None:
    route = respx.get(f"{API}/package_show").respond(503)
    with httpx.Client() as client, pytest.raises(SourceError, match="503"):
        neso_demand.list_resources(client, settings)
    assert route.call_count == settings.http_max_retries


@respx.mock
def test_ingest_update_drops_forecasts_and_stores(settings: Settings) -> None:
    csv = (
        b"SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,TSD,FORECAST_ACTUAL_INDICATOR\n"
        b"2026-09-20,1,20000,22000,A\n"
        b"2026-09-20,2,20100,22100,A\n"
        b"2026-09-23,1,,,F\n"
    )
    respx.get(neso_demand.UPDATE_URL).respond(content=csv)
    with httpx.Client() as client:
        rows, bad = neso_demand.ingest_update(client, settings)
    assert (rows, bad) == (2, 0)
    stored = pd.read_parquet(settings.processed_dir / neso_demand.UPDATE_SOURCE / "data.parquet")
    assert stored["nd_mw"].tolist() == [20000, 20100]


@respx.mock
def test_ingest_end_to_end_writes_processed_parquet(settings: Settings) -> None:
    respx.get(f"{API}/package_show").respond(
        json=json.loads((FIXTURES / "neso_package_show.json").read_text())
    )
    respx.get(url__regex=r".*demanddata_2024\.csv").respond(content=make_csv(["2024-06-01"]))

    with httpx.Client() as client:
        reports = neso_demand.ingest(client, settings, start_year=2024, end_year=2024)

    assert reports[0].rows == 48
    stored = pd.read_parquet(neso_demand.processed_path(settings, 2024))
    assert len(stored) == 48
    assert str(stored["timestamp_utc"].dt.tz) == "UTC"
    assert neso_demand.raw_path(settings, 2024).exists()


@respx.mock
def test_ingest_skips_download_for_old_years_already_on_disk(settings: Settings) -> None:
    respx.get(f"{API}/package_show").respond(
        json=json.loads((FIXTURES / "neso_package_show.json").read_text())
    )
    csv_route = respx.get(url__regex=r".*demanddata_2024\.csv").respond(
        content=make_csv(["2024-06-01"])
    )
    with httpx.Client() as client:
        neso_demand.ingest(client, settings, 2024, 2024)
        second = neso_demand.ingest(client, settings, 2024, 2024)

    assert csv_route.call_count == 1
    assert second[0].downloaded is False
