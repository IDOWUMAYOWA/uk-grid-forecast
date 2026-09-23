"""Load cleaned datasets from local storage.

The rest of the project (features, training, monitoring) reads data only
through these functions, never by opening files directly. If storage moves
(for example to a cloud bucket), only this module changes.
"""

import pandas as pd

from gridcast.config import Settings
from gridcast.ingest import elexon_generation, neso_demand, weather
from gridcast.storage import read_parquet_dir


def load_demand(settings: Settings) -> pd.DataFrame:
    """NESO national demand: yearly history, extended to today by the update file.

    Where both sources cover the same half hour, the yearly file wins,
    because NESO revises recent values before they reach the yearly file.
    """
    history = read_parquet_dir(settings.processed_dir / neso_demand.SOURCE)
    update_file = settings.processed_dir / neso_demand.UPDATE_SOURCE / "data.parquet"
    if not update_file.exists():
        return history.sort_values("timestamp_utc").reset_index(drop=True)

    update = pd.read_parquet(update_file)
    newer = update[~update["timestamp_utc"].isin(history["timestamp_utc"])]
    combined = pd.concat([history, newer], ignore_index=True)
    return combined.sort_values("timestamp_utc").reset_index(drop=True)


def load_generation(settings: Settings) -> pd.DataFrame:
    """Elexon generation by fuel type, long format."""
    return read_parquet_dir(settings.processed_dir / elexon_generation.SOURCE)


def load_weather(settings: Settings) -> pd.DataFrame:
    """Hourly weather forecasts for every location, long format."""
    return read_parquet_dir(settings.processed_dir / weather.SOURCE)
