import pandas as pd
import pytest

from gridcast.features.weather import (
    HEATING_BASE_C,
    add_weather_features,
    national_weather,
    to_half_hourly,
)
from gridcast.ingest.weather import HOURLY_VARIABLES, LOCATIONS
from tests.conftest import make_demand, make_weather


def test_national_weather_is_population_weighted() -> None:
    weather = make_weather(days=2)
    national = national_weather(weather)
    at_noon = pd.Timestamp("2025-01-01 12:00", tz="UTC")

    city_values = weather[weather["timestamp_utc"] == at_noon].set_index("location")
    temperatures = city_values["temperature_2m"].astype(float)
    expected = sum(loc.weight * float(temperatures.loc[loc.name]) for loc in LOCATIONS)
    assert national.loc[at_noon, "temperature_2m"] == pytest.approx(expected)


def test_missing_city_does_not_drag_the_average_down() -> None:
    weather = make_weather(days=2)
    weather.loc[weather["location"] == "glasgow", "temperature_2m"] = None
    national = national_weather(weather)
    assert national["temperature_2m"].notna().all()
    assert national["temperature_2m"].between(-10, 30).all()


def test_unknown_location_is_rejected() -> None:
    weather = make_weather(days=1)
    weather.loc[0, "location"] = "atlantis"
    with pytest.raises(ValueError, match="atlantis"):
        national_weather(weather)


def test_hourly_weather_is_interpolated_to_half_hours() -> None:
    hourly = pd.DataFrame(
        {"temperature_2m": [10.0, 12.0]},
        index=pd.DatetimeIndex(["2025-01-01 00:00", "2025-01-01 01:00"], tz="UTC"),
    )
    index = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=3, freq="30min", tz="UTC"))
    result = to_half_hourly(hourly, index)
    assert result["temperature_2m"].tolist() == [10.0, 11.0, 12.0]


def test_degree_days_and_thermal_inertia() -> None:
    df = add_weather_features(make_demand(days=5), make_weather(days=6))
    assert (df["heating_degrees"] >= 0).all()
    assert (df["cooling_degrees"] >= 0).all()
    # Below the heating threshold, heating degrees must be positive.
    cold = df[df["temperature_2m"] < HEATING_BASE_C]
    assert (cold["heating_degrees"] > 0).all()
    # Smoothed temperature is less volatile than the raw series.
    assert df["temperature_smoothed"].std() < df["temperature_2m"].std()
    assert set(HOURLY_VARIABLES) - {"precipitation"} <= set(df.columns)
