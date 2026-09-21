"""GB electricity settlement time.

The GB market splits each day into half-hour "settlement periods" numbered
from 1, starting at local midnight (Europe/London). Because of daylight
saving, days are not always 48 periods long:

- the spring clock change day has 46 periods (clocks jump forward an hour)
- the autumn clock change day has 50 periods (an hour is repeated)

Local times are therefore ambiguous, so we convert everything to UTC, where
every half hour is unique, and only use local time for display.
"""

import pandas as pd

LONDON = "Europe/London"
HALF_HOUR = pd.Timedelta(minutes=30)


def periods_in_day(settlement_date: pd.Series) -> pd.Series:
    """Number of settlement periods (46, 48 or 50) for each date."""
    day = pd.to_datetime(settlement_date).dt.normalize()
    start = day.dt.tz_localize(LONDON)
    end = (day + pd.Timedelta(days=1)).dt.tz_localize(LONDON)
    return ((end - start) / HALF_HOUR).astype("int64")


def to_utc(settlement_date: pd.Series, settlement_period: pd.Series) -> pd.Series:
    """UTC start time of each (settlement date, settlement period) pair.

    Local midnight is never ambiguous in the UK (clocks change at 01:00 and
    02:00), so we find midnight in UTC and add 30 minutes per period. This
    gives the right answer on 46- and 50-period days too.
    """
    midnight_utc = (
        pd.to_datetime(settlement_date).dt.normalize().dt.tz_localize(LONDON).dt.tz_convert("UTC")
    )
    offset = (settlement_period.astype("int64") - 1) * HALF_HOUR
    return (midnight_utc + offset).dt.as_unit("ns")
