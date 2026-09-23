"""Calendar features.

Electricity demand follows human routines, so the calendar explains most of
the variation before weather is even considered: time of day, weekday versus
weekend, season, and public holidays (a bank holiday Monday looks like a
Sunday, not a Monday).

Cyclical values (time of day, day of year) are encoded as sine and cosine
pairs so that the model knows 23:30 is next to 00:00, and 31 December is
next to 1 January. A plain number would put them far apart.
"""

import numpy as np
import pandas as pd
from holidays import country_holidays

from gridcast.settlement import periods_in_day

# Great Britain only: NESO's demand data excludes Northern Ireland.
_GB_SUBDIVISIONS = ("ENG", "SCT")


def _cyclical(values: pd.Series, period: float, name: str) -> pd.DataFrame:
    angle = 2 * np.pi * values / period
    return pd.DataFrame({f"{name}_sin": np.sin(angle), f"{name}_cos": np.cos(angle)})


def gb_holidays(years: list[int]) -> dict[str, set[pd.Timestamp]]:
    """Bank holiday dates per GB nation (England/Wales and Scotland differ)."""
    out: dict[str, set[pd.Timestamp]] = {}
    for subdiv in _GB_SUBDIVISIONS:
        days = country_holidays("GB", subdiv=subdiv, years=years)
        out[subdiv] = {pd.Timestamp(d) for d in days}
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features. Requires settlement_date and settlement_period."""
    out = df.copy()
    date = out["settlement_date"]
    period = out["settlement_period"]

    out["day_of_week"] = date.dt.dayofweek.astype("int64")
    out["month"] = date.dt.month.astype("int64")
    out["day_of_year"] = date.dt.dayofyear.astype("int64")
    out["year"] = date.dt.year.astype("int64")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int64")

    # Position through the day as a fraction, so 46/48/50-period days align.
    out["day_fraction"] = (period - 1) / periods_in_day(date)
    out = pd.concat([out, _cyclical(out["day_fraction"], 1.0, "time_of_day")], axis=1)
    out = pd.concat([out, _cyclical(out["day_of_year"], 365.25, "day_of_year")], axis=1)

    years = sorted(out["year"].unique().tolist())
    holidays_by_nation = gb_holidays(years)
    eng = holidays_by_nation["ENG"]
    sct = holidays_by_nation["SCT"]
    out["is_holiday_eng_wales"] = date.isin(eng).astype("int64")
    out["is_holiday_scotland"] = date.isin(sct).astype("int64")

    # A working day that sits between a holiday and a weekend behaves oddly,
    # so tell the model how close the nearest holiday is.
    all_holidays = pd.Series(sorted(eng | sct))
    if len(all_holidays):
        gaps = np.abs(date.to_numpy()[:, None] - all_holidays.to_numpy()[None, :]).min(axis=1)
        days = gaps / np.timedelta64(1, "D")
        out["days_to_nearest_holiday"] = pd.Series(days, index=out.index).clip(upper=14)
    else:
        out["days_to_nearest_holiday"] = 14.0

    # The week between Christmas and New Year has its own demand pattern.
    out["is_christmas_period"] = (
        ((date.dt.month == 12) & (date.dt.day >= 24)) | ((date.dt.month == 1) & (date.dt.day <= 1))
    ).astype("int64")

    return out


def solar_elevation(
    timestamp_utc: pd.Series, latitude: float = 54.0, longitude: float = -2.0
) -> pd.Series:
    """Approximate solar elevation angle (degrees) at the centre of GB.

    Drives both lighting demand and rooftop-solar output, and unlike a plain
    clock time it shifts correctly through the year.
    """
    ts = pd.DatetimeIndex(timestamp_utc)
    day_of_year = ts.dayofyear.to_numpy()
    hours = ts.hour.to_numpy() + ts.minute.to_numpy() / 60

    # Declination of the sun and the hour angle (standard approximations).
    declination = np.radians(-23.44 * np.cos(2 * np.pi * (day_of_year + 10) / 365.25))
    hour_angle = np.radians(15 * (hours + longitude / 15 - 12))
    lat = np.radians(latitude)

    sin_elevation = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(
        hour_angle
    )
    return pd.Series(
        np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1))), index=timestamp_utc.index
    )
