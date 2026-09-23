"""Feature table assembly, including the leakage guarantee."""

from pathlib import Path

import pandas as pd
import pytest

from gridcast.config import Settings
from gridcast.errors import DataQualityError
from gridcast.features.build import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    LEAKY_COLUMNS,
    TARGET,
    build_and_store,
    build_features,
)
from gridcast.ingest import neso_demand
from gridcast.storage import write_parquet_atomic
from tests.conftest import make_demand, make_weather


@pytest.fixture
def features() -> pd.DataFrame:
    return build_features(make_demand(days=60), make_weather(days=61))


def test_columns_are_unique(features: pd.DataFrame) -> None:
    """Duplicated columns are legal in pandas but break Parquet on write."""
    assert features.columns.is_unique


def test_table_round_trips_through_parquet(features: pd.DataFrame, tmp_path: Path) -> None:
    """Building a table is not enough; it has to be storable and readable."""
    path = tmp_path / "features.parquet"
    write_parquet_atomic(features, path)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(features, restored)


def test_feature_table_shape_and_completeness(features: pd.DataFrame) -> None:
    assert not features.empty
    assert features[FEATURE_COLUMNS].notna().all().all()
    assert features["timestamp_utc"].is_unique
    assert features["timestamp_utc"].is_monotonic_increasing


def test_rows_without_two_weeks_of_history_are_dropped(features: pd.DataFrame) -> None:
    assert features["settlement_date"].min() == pd.Timestamp("2025-01-15")


def test_target_is_present_but_not_a_feature(features: pd.DataFrame) -> None:
    assert TARGET in features.columns
    assert TARGET not in FEATURE_COLUMNS
    assert set(IDENTIFIER_COLUMNS) <= set(features.columns)


def test_same_time_measurements_are_never_features() -> None:
    assert not (LEAKY_COLUMNS & set(FEATURE_COLUMNS))


def test_missing_weather_is_reported_clearly() -> None:
    weather = make_weather(days=61)
    with pytest.raises(ValueError):
        build_features(make_demand(days=60), weather.assign(location="mars"))


def test_too_little_history_produces_an_empty_table_not_wrong_rows() -> None:
    result = build_features(make_demand(days=1), make_weather(days=1))
    assert result.empty


def test_build_and_store_refuses_to_save_an_empty_table(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    write_parquet_atomic(make_demand(days=1), neso_demand.processed_path(settings, 2025))
    write_parquet_atomic(make_weather(days=1), settings.processed_dir / "open_meteo" / "a.parquet")
    with pytest.raises(DataQualityError, match="empty"):
        build_and_store(settings)


# ------------------------------------------------------------------ leakage


def test_features_cannot_see_the_target_day_or_the_day_before() -> None:
    """The forecast for day D is issued the morning of D-1.

    So the features for day D must be identical whether or not demand on
    D-1 and D exists. If this test fails, the model is cheating: offline
    scores would look great and production would be far worse.
    """
    demand = make_demand(days=60)
    weather = make_weather(days=61)
    target_day = pd.Timestamp("2025-02-20")

    honest = build_features(demand, weather)

    # Replace demand on the target day and the day before with nonsense.
    poisoned_demand = demand.copy()
    recent = poisoned_demand["settlement_date"] >= target_day - pd.Timedelta(days=1)
    poisoned_demand.loc[recent, "nd_mw"] = 999_999.0
    poisoned = build_features(poisoned_demand, weather)

    honest_day = honest[honest["settlement_date"] == target_day][FEATURE_COLUMNS]
    poisoned_day = poisoned[poisoned["settlement_date"] == target_day][FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(honest_day, poisoned_day)


def test_poisoning_older_data_does_change_features() -> None:
    """Sanity check: the leakage test above would actually catch a mistake."""
    demand = make_demand(days=60)
    weather = make_weather(days=61)
    target_day = pd.Timestamp("2025-02-20")

    honest = build_features(demand, weather)
    poisoned_demand = demand.copy()
    older = poisoned_demand["settlement_date"] == target_day - pd.Timedelta(days=7)
    poisoned_demand.loc[older, "nd_mw"] = 999_999.0
    poisoned = build_features(poisoned_demand, weather)

    honest_day = honest[honest["settlement_date"] == target_day]["nd_lag_7d_mw"]
    poisoned_day = poisoned[poisoned["settlement_date"] == target_day]["nd_lag_7d_mw"]
    assert not honest_day.equals(poisoned_day)
