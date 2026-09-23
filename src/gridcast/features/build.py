"""Assemble the model-ready feature table.

    demand + weather ──> calendar ──> lags ──> weather ──> features.parquet

The output has one row per half hour, with the target (`nd_mw`) and every
feature the model is allowed to use. Nothing in it depends on information
that would be unavailable when a day-ahead forecast is issued.
"""

from pathlib import Path

import pandas as pd

from gridcast.config import Settings
from gridcast.data import load_demand, load_weather
from gridcast.errors import DataQualityError
from gridcast.features.calendar import add_calendar_features, solar_elevation
from gridcast.features.lags import add_lag_features
from gridcast.features.weather import add_weather_features
from gridcast.logging import get_logger
from gridcast.storage import write_parquet_atomic

log = get_logger(__name__)

TARGET = "nd_mw"

# Identify each row. `settlement_period` is also a feature, so the two lists
# overlap; keep them de-duplicated when assembling the table.
IDENTIFIER_COLUMNS = ["timestamp_utc", "settlement_date", "settlement_period", TARGET]

FEATURE_COLUMNS = [
    # calendar
    "settlement_period",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "day_fraction",
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_holiday_eng_wales",
    "is_holiday_scotland",
    "days_to_nearest_holiday",
    "is_christmas_period",
    "solar_elevation",
    # weather
    "temperature_2m",
    "apparent_temperature",
    "heating_degrees",
    "cooling_degrees",
    "temperature_smoothed",
    "temperature_change_24h",
    "wind_speed_10m",
    "wind_speed_cubed",
    "cloud_cover",
    "shortwave_radiation",
    "relative_humidity_2m",
    # lagged demand
    "nd_lag_2d_mw",
    "nd_lag_3d_mw",
    "nd_lag_7d_mw",
    "nd_lag_14d_mw",
    "nd_lag_week_mean_mw",
    "nd_prev_day_mean_mw",
    "nd_prev_day_max_mw",
    "nd_prev_day_min_mw",
]

# Columns that must never be used as features: they are measured at the same
# moment as the target, so they would not exist when forecasting ahead.
LEAKY_COLUMNS = frozenset(
    {
        "tsd_mw",
        "england_wales_demand_mw",
        "embedded_wind_generation_mw",
        "embedded_solar_generation_mw",
        "non_bm_stor_mw",
        "pump_storage_pumping_mw",
        "scottish_transfer_mw",
    }
)


def features_path(settings: Settings) -> Path:
    return settings.data_dir / "features" / "features.parquet"


def build_features(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Build the feature table from cleaned demand and weather data."""
    base = demand[["timestamp_utc", "settlement_date", "settlement_period", TARGET]].copy()

    out = add_calendar_features(base)
    out["solar_elevation"] = solar_elevation(out["timestamp_utc"])
    out = add_lag_features(out)
    out = add_weather_features(out, weather)

    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise DataQualityError(f"Feature build produced no column for: {missing}")
    leaked = LEAKY_COLUMNS & set(FEATURE_COLUMNS)
    if leaked:  # pragma: no cover - guards against a future editing mistake
        raise DataQualityError(f"Leaky columns present in FEATURE_COLUMNS: {sorted(leaked)}")

    columns = IDENTIFIER_COLUMNS + [c for c in FEATURE_COLUMNS if c not in IDENTIFIER_COLUMNS]
    out = out[columns]

    # The earliest rows have no two-week history, so their lags are empty.
    before = len(out)
    out = out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    log.info("features.built", rows=len(out), dropped_incomplete=before - len(out))
    return out


def build_and_store(settings: Settings) -> pd.DataFrame:
    """Load stored data, build features, and save them for training."""
    demand = load_demand(settings)
    weather = load_weather(settings)
    features = build_features(demand, weather)
    if features.empty:
        raise DataQualityError("Feature table is empty; check that ingestion has run.")
    write_parquet_atomic(features, features_path(settings))
    return features
