# ADR 0004: Features must respect day-ahead availability

**Status:** Accepted
**Date:** 2026-09-22

## Context

We forecast every half hour of day D+1, issuing the forecast on the morning
of day D (NESO refreshes its demand file by about 08:30 UTC).

At that moment the most recent *complete* day of demand is D-1. For the last
half hour of D+1, the newest usable observation is therefore up to 38.5 hours
old. The textbook feature "demand at the same time yesterday" (a lag of 48
half hours) does not exist yet for most of the horizon. Using it would inflate
offline accuracy and collapse in production.

## Decision

1. Lagged demand features use offsets of **2 days or more** only: same period
   2, 3, 7 and 14 days earlier, plus whole-day summaries of D-2.
2. Values measured at the same instant as the target (TSD, England & Wales
   demand, embedded wind/solar outturn, interconnector flows, pump storage)
   are listed in `LEAKY_COLUMNS` and excluded from features.
3. Weather features come from archived *forecasts* (ADR 0003), so they are
   legitimately available ahead of time.
4. Lags are matched on (settlement date, settlement period), not clock time,
   so they remain correct across 46- and 50-period days. Where no matching
   period exists a week earlier, the lag is null rather than approximated.
5. A test (`test_features_cannot_see_the_target_day_or_the_day_before`)
   corrupts demand for D-1 and D and asserts that every feature for day D is
   unchanged. A second test confirms that corrupting older data *does* change
   features, so the first test cannot pass trivially.

## Consequences

- Accuracy will be lower than published "day-ahead" results that quietly use
  48-period lags, but it is achievable in production.
- Intraday forecasting (shorter horizons) would use a different feature set;
  if added later it needs its own availability rules, not a relaxation of
  these.
