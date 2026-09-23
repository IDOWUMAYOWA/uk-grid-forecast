import numpy as np
import pandas as pd

from gridcast.features.calendar import add_calendar_features, solar_elevation
from tests.conftest import make_demand


def test_holidays_and_weekends_are_flagged() -> None:
    df = add_calendar_features(make_demand("2025-12-20", days=14))
    by_date = df.groupby("settlement_date")[
        ["is_weekend", "is_holiday_eng_wales", "is_christmas_period"]
    ].max()

    assert by_date.loc["2025-12-25", "is_holiday_eng_wales"] == 1  # Christmas Day
    assert by_date.loc["2025-12-26", "is_holiday_eng_wales"] == 1  # Boxing Day
    assert by_date.loc["2025-12-23", "is_holiday_eng_wales"] == 0
    assert by_date.loc["2025-12-21", "is_weekend"] == 1  # Sunday
    assert by_date.loc["2025-12-28", "is_christmas_period"] == 1


def test_scotland_has_its_own_holiday() -> None:
    df = add_calendar_features(make_demand("2026-01-01", days=3))
    jan2 = df[df["settlement_date"] == "2026-01-02"]
    assert jan2["is_holiday_scotland"].max() == 1  # 2 January is Scottish only
    assert jan2["is_holiday_eng_wales"].max() == 0


def test_cyclical_encoding_wraps_around_midnight() -> None:
    df = add_calendar_features(make_demand("2025-06-01", days=2))
    day = df[df["settlement_date"] == "2025-06-01"]
    first, last = day.iloc[0], day.iloc[-1]
    # 23:30 and 00:00 must be neighbours in the encoded space.
    distance = np.hypot(
        first["time_of_day_sin"] - last["time_of_day_sin"],
        first["time_of_day_cos"] - last["time_of_day_cos"],
    )
    assert distance < 0.2


def test_day_fraction_handles_clock_change_days() -> None:
    df = add_calendar_features(make_demand("2025-10-25", days=2))
    long_day = df[df["settlement_date"] == "2025-10-26"]  # 50 periods
    assert len(long_day) == 50
    assert long_day["day_fraction"].max() < 1.0
    assert long_day["day_fraction"].is_monotonic_increasing


def test_solar_elevation_is_higher_at_noon_and_in_summer() -> None:
    times = pd.Series(
        pd.to_datetime(["2025-06-21 12:00", "2025-06-21 00:00", "2025-12-21 12:00"], utc=True)
    )
    elevation = solar_elevation(times)
    assert elevation.iloc[0] > elevation.iloc[2] > 0  # summer noon > winter noon > 0
    assert elevation.iloc[1] < 0  # sun is below the horizon at midnight
