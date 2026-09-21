from datetime import date

import pandas as pd
import pytest

from gridcast.settlement import periods_in_day, to_utc


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 1, 15), 48),  # ordinary winter day
        (date(2026, 7, 15), 48),  # ordinary summer day
        (date(2026, 3, 29), 46),  # spring forward: clocks skip 01:00-02:00
        (date(2026, 10, 25), 50),  # fall back: 01:00-02:00 happens twice
    ],
)
def test_periods_in_day(day: date, expected: int) -> None:
    assert periods_in_day(pd.Series([pd.Timestamp(day)])).iloc[0] == expected


def test_winter_period_1_is_midnight_utc() -> None:
    ts = to_utc(pd.Series([pd.Timestamp("2026-01-15")]), pd.Series([1]))
    assert ts.iloc[0] == pd.Timestamp("2026-01-15 00:00", tz="UTC")


def test_summer_period_1_is_2300_utc_previous_day() -> None:
    # Local midnight in BST (UTC+1) is 23:00 UTC the day before.
    ts = to_utc(pd.Series([pd.Timestamp("2026-07-15")]), pd.Series([1]))
    assert ts.iloc[0] == pd.Timestamp("2026-07-14 23:00", tz="UTC")


def test_autumn_clock_change_day_has_no_duplicate_utc_times() -> None:
    day = pd.Series([pd.Timestamp("2026-10-25")] * 50)
    ts = to_utc(day, pd.Series(range(1, 51)))
    assert ts.is_unique
    assert ts.is_monotonic_increasing
    assert (ts.diff().dropna() == pd.Timedelta(minutes=30)).all()


def test_spring_clock_change_day_is_continuous() -> None:
    day = pd.Series([pd.Timestamp("2026-03-29")] * 46)
    ts = to_utc(day, pd.Series(range(1, 47)))
    assert ts.iloc[-1] + pd.Timedelta(minutes=30) == pd.Timestamp("2026-03-29 23:00", tz="UTC")
