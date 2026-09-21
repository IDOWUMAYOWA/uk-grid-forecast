"""Hits the real NESO API. Run with: make test-all"""

from pathlib import Path

import pytest

from gridcast.config import Settings
from gridcast.http import make_client
from gridcast.ingest import neso_demand

pytestmark = pytest.mark.integration


def test_live_catalogue_lists_recent_years(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    with make_client(settings) as client:
        resources = neso_demand.list_resources(client, settings)
    assert {2022, 2023, 2024} <= set(resources)


def test_live_ingest_one_complete_year(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    with make_client(settings) as client:
        reports = neso_demand.ingest(client, settings, 2023, 2023)
    # 2023: 365 days x 48 periods (the two clock-change days net out to zero).
    # Allow a little slack for gaps in the source data.
    expected = 365 * 48
    assert reports[0].rows >= 0.99 * expected
    assert reports[0].rows + reports[0].quarantined <= expected
