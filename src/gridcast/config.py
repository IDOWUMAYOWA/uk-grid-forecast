"""Centralised, typed configuration.

All settings are read from environment variables prefixed with ``GRIDCAST_``
(or a local ``.env`` file). Nothing else in the codebase should read
``os.environ`` directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GRIDCAST_",
        extra="ignore",
    )

    env: Literal["dev", "ci", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = Field(default=False, description="Emit JSON logs (use in prod).")

    data_dir: Path = Path("data")

    # External data sources (all public, no API key required)
    neso_api_base: str = "https://api.neso.energy/api/3/action"
    elexon_api_base: str = "https://data.elexon.co.uk/bmrs/api/v1"
    open_meteo_forecast_base: str = "https://historical-forecast-api.open-meteo.com/v1"

    http_timeout_s: float = 30.0
    http_max_retries: int = 3
    http_backoff_s: float = Field(default=1.0, description="Base delay for exponential backoff.")

    # Data quality: stop the pipeline if more than this share of rows is invalid
    max_invalid_row_fraction: float = 0.01

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
