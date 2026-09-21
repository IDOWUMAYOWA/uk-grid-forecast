# ADR 0002: Data ingestion layout and data contracts

**Status:** Accepted
**Date:** 2026-09-21

## Context

The model is only as good as its input data. Source data from NESO changes
format between years, mixes actuals with forecasts in the current-year file,
contains occasional duplicates, and has known gaps. GB settlement periods
also produce 46- and 50-period days at clock changes.

## Decision

1. **Three storage zones.**
   - `data/raw/` holds files exactly as downloaded, never modified, so any
     cleaning change can be replayed without re-downloading.
   - `data/processed/` holds cleaned, validated Parquet, partitioned by year.
   - `data/quarantine/` holds rows that failed validation, for inspection.
2. **UTC everywhere.** Every half hour is keyed by `timestamp_utc`, derived
   from (settlement date, settlement period) via local midnight. Local time is
   for display only.
3. **Data contracts with Pandera.** Structural failures (missing columns,
   wrong types) stop the pipeline. Row-level failures are quarantined unless
   they exceed `max_invalid_row_fraction` (default 1%), which also stops it.
4. **Incremental downloads.** Complete past years are fetched once; the
   current and previous year are always refreshed because NESO revises them.
5. **Retries with exponential backoff** for transient errors (timeouts, 429,
   5xx) only; permanent errors such as 404 fail immediately.
6. **Offline tests.** Unit tests mock HTTP with `respx`; tests that hit real
   APIs are marked `integration` and excluded from CI.

## Consequences

Bad upstream data is detected on the day it appears rather than weeks later
as a mysterious drop in model accuracy. The quarantine threshold may need
tuning once we observe real failure rates.
