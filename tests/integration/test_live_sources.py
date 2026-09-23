"""Hits the real Elexon and Open-Meteo APIs. Run with: make test-all"""

from datetime import date
from pathlib import Path

import pytest

from gridcast.config import Settings
from gridcast.http import make_client
from gridcast.ingest import elexon_generation, neso_demand, weather

pytestmark = pytest.mark.integration


def test_live_neso_update(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    with make_client(settings) as client:
        rows, _ = neso_demand.ingest_update(client, settings)
    assert rows > 28 * 48  # at least all of last month


def test_live_elexon_one_month(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    with make_client(settings) as client:
        reports = elexon_generation.ingest(client, settings, date(2024, 6, 1), date(2024, 6, 30))
    assert reports[0].rows > 30 * 48 * 10  # 30 days x 48 periods x 10+ fuel types


def test_live_weather_one_year(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    with make_client(settings) as client:
        reports = weather.ingest(client, settings, 2024, 2024)
    assert all(r.rows >= 0.99 * 366 * 24 for r in reports)
