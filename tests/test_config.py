from pathlib import Path

import pytest

from gridcast.config import get_settings


def test_defaults() -> None:
    settings = get_settings()
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.elexon_api_base.startswith("https://")


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GRIDCAST_ENV", "prod")
    monkeypatch.setenv("GRIDCAST_DATA_DIR", str(tmp_path))
    settings = get_settings()
    assert settings.env == "prod"
    assert settings.data_dir == tmp_path


def test_invalid_env_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRIDCAST_ENV", "staging-typo")
    with pytest.raises(ValueError):
        get_settings()
