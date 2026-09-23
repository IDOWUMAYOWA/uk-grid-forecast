"""Lagged demand features, built so they can never see the future.

The hardest part of forecasting is not the model, it is being honest about
what is known when the forecast is made.

We issue a day-ahead forecast at a fixed time each morning (09:00 UTC by
default) for every half hour of the following day. At that moment the latest
complete day of demand is *yesterday*. So for a target half hour tomorrow,
the most recent usable observation can be up to 38.5 hours old.

That rules out the obvious "demand at the same time yesterday" feature: for
tomorrow evening, that value has not happened yet at the moment we forecast.
Using it would make offline results look excellent and production results
fall apart. Every feature here is therefore built from:

- the same half hour two days before the target (always complete), and
- the same half hour one week before (captures weekly rhythm), and
- whole-day summaries of the last complete day.

Lags are aligned by (settlement date, settlement period) rather than by
clock time, so they stay correct across clock-change days.
"""

import pandas as pd

# Offsets in days, relative to the target's settlement date.
SAME_PERIOD_LAG_DAYS = (2, 3, 7, 14)
DAILY_SUMMARY_LAG_DAYS = 2


def _same_period_lag(df: pd.DataFrame, days: int) -> pd.Series:
    """Demand at the same settlement period, `days` days earlier."""
    source = df[["settlement_date", "settlement_period", "nd_mw"]].copy()
    source["settlement_date"] = source["settlement_date"] + pd.Timedelta(days=days)
    source = source.rename(columns={"nd_mw": f"nd_lag_{days}d_mw"})
    merged = df.merge(source, on=["settlement_date", "settlement_period"], how="left")
    return merged[f"nd_lag_{days}d_mw"]


def _daily_summaries(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Mean, peak and minimum demand of a complete earlier day."""
    grouped = df.groupby("settlement_date")["nd_mw"]
    daily = pd.DataFrame(
        {
            "nd_prev_day_mean_mw": grouped.mean(),
            "nd_prev_day_max_mw": grouped.max(),
            "nd_prev_day_min_mw": grouped.min(),
        }
    )
    daily.index = daily.index + pd.Timedelta(days=days)
    return daily


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged demand features. Requires settlement_date, period, nd_mw."""
    out = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    for days in SAME_PERIOD_LAG_DAYS:
        out[f"nd_lag_{days}d_mw"] = _same_period_lag(out, days)

    summaries = _daily_summaries(out, DAILY_SUMMARY_LAG_DAYS)
    out = out.merge(summaries, left_on="settlement_date", right_index=True, how="left")

    # Weekly average of this half hour, from the last two complete weeks.
    out["nd_lag_week_mean_mw"] = out[["nd_lag_7d_mw", "nd_lag_14d_mw"]].mean(axis=1)

    return out
