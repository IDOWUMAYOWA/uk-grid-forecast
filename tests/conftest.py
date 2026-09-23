from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from gridcast.config import get_settings
from gridcast.ingest.weather import HOURLY_VARIABLES, LOCATIONS
from gridcast.settlement import periods_in_day, to_utc


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Settings are cached; reset between tests so env overrides take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_demand(start: str = "2025-01-01", days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic half-hourly demand with a daily shape and weekly rhythm."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range(start, periods=days, freq="D"):
        for period in range(1, int(periods_in_day(pd.Series([day])).iloc[0]) + 1):
            hour = (period - 1) / 2
            shape = 25_000 + 6_000 * np.sin((hour - 6) / 24 * 2 * np.pi)
            weekend = -2_000 if day.dayofweek >= 5 else 0
            rows.append((day, period, shape + weekend + rng.normal(0, 150)))

    df = pd.DataFrame(rows, columns=["settlement_date", "settlement_period", "nd_mw"])
    df["timestamp_utc"] = to_utc(df["settlement_date"], df["settlement_period"])
    return df


def make_weather(start: str = "2025-01-01", days: int = 61, seed: int = 1) -> pd.DataFrame:
    """Synthetic hourly weather for every ingested location."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    frames = []
    for i, location in enumerate(LOCATIONS):
        hours = times.hour.to_numpy()
        data = {
            "temperature_2m": 8 + 4 * np.sin((hours - 9) / 24 * 2 * np.pi) - i * 0.5,
            "apparent_temperature": 6 + 4 * np.sin((hours - 9) / 24 * 2 * np.pi),
            "wind_speed_10m": np.abs(rng.normal(6, 2, len(times))),
            "cloud_cover": rng.uniform(0, 100, len(times)),
            "shortwave_radiation": np.clip(400 * np.sin((hours - 6) / 12 * np.pi), 0, None),
            "relative_humidity_2m": rng.uniform(50, 95, len(times)),
            "precipitation": np.zeros(len(times)),
        }
        frame = pd.DataFrame({v: data[v] for v in HOURLY_VARIABLES})
        frame.insert(0, "timestamp_utc", times)
        frame.insert(1, "location", location.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
