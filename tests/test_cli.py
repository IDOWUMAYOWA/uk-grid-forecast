import json
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from gridcast import __version__
from gridcast.cli import app
from tests.test_neso_demand import API, FIXTURES, make_csv

runner = CliRunner()


def test_info_command() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


@respx.mock
def test_ingest_neso_demand_prints_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GRIDCAST_DATA_DIR", str(tmp_path))
    respx.get(f"{API}/package_show").respond(
        json=json.loads((FIXTURES / "neso_package_show.json").read_text())
    )
    respx.get(url__regex=r".*demanddata_2024\.csv").respond(content=make_csv(["2024-06-01"]))

    result = runner.invoke(
        app, ["ingest", "neso-demand", "--start-year", "2024", "--end-year", "2024"]
    )
    assert result.exit_code == 0, result.stdout
    assert "2024" in result.stdout
    assert "48" in result.stdout


@respx.mock
def test_ingest_exits_non_zero_when_source_is_broken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GRIDCAST_DATA_DIR", str(tmp_path))
    respx.get(f"{API}/package_show").respond(json={"success": False, "error": "down"})

    result = runner.invoke(app, ["ingest", "neso-demand", "--start-year", "2024"])
    assert result.exit_code == 1


def test_status_reports_missing_datasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GRIDCAST_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "not ingested yet" in result.stdout
