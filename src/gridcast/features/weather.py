"""Weather features.

Weather is the strongest driver of demand after the calendar. Three ideas
matter here:

1. **National, not local.** Seven cities are combined into one GB-wide value,
   weighted by population, because demand responds to weather where people
   are.
2. **Degree days, not raw temperature.** Demand responds non-linearly:
   below about 15.5C people heat, above about 18C some cool. Heating and
   cooling degree days express this directly, so even a linear model can
   use it.
3. **Buildings have memory.** After a cold night a house stays cold, so
   yesterday's temperature still affects today's demand. An exponentially
   weighted average of recent temperature captures this thermal inertia.

Weather arrives hourly and demand is half-hourly, so values are interpolated
onto the half-hourly grid.
"""

import pandas as pd

from gridcast.ingest.weather import LOCATIONS

# Typical UK thresholds used by the energy industry.
HEATING_BASE_C = 15.5
COOLING_BASE_C = 18.0

WEATHER_VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "relative_humidity_2m",
)


def national_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-city weather into one population-weighted GB series."""
    weights = {loc.name: loc.weight for loc in LOCATIONS}
    df = weather.copy()
    df["weight"] = df["location"].map(weights)
    if df["weight"].isna().any():
        unknown = sorted(set(df.loc[df["weight"].isna(), "location"]))
        raise ValueError(f"Weather data for unknown locations: {unknown}")

    out = {}
    for variable in WEATHER_VARIABLES:
        weighted = df[variable] * df["weight"]
        grouped = weighted.groupby(df["timestamp_utc"]).sum()
        # Divide by the weight actually present, so a missing city doesn't
        # silently drag the national average down.
        present = df.loc[df[variable].notna(), "weight"].groupby(df["timestamp_utc"]).sum()
        out[variable] = grouped / present

    return pd.DataFrame(out).sort_index()


def to_half_hourly(hourly: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate hourly weather onto the half-hourly demand timestamps."""
    combined = hourly.reindex(hourly.index.union(index)).interpolate(method="time", limit=2)
    return combined.reindex(index)


def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join national weather onto the half-hourly grid and derive features."""
    national = national_weather(weather)
    index = pd.DatetimeIndex(df["timestamp_utc"])
    aligned = to_half_hourly(national, index).reset_index(drop=True)

    out = pd.concat([df.reset_index(drop=True), aligned], axis=1)

    temp = out["temperature_2m"]
    out["heating_degrees"] = (HEATING_BASE_C - temp).clip(lower=0)
    out["cooling_degrees"] = (temp - COOLING_BASE_C).clip(lower=0)

    # Thermal inertia: half-life of one day over the half-hourly series.
    out["temperature_smoothed"] = temp.ewm(halflife=48, ignore_na=True).mean()
    out["temperature_change_24h"] = temp - temp.shift(48)

    # Wind chill effect on heating demand, and wind's effect on wind output.
    out["wind_speed_cubed"] = out["wind_speed_10m"] ** 3

    return out
