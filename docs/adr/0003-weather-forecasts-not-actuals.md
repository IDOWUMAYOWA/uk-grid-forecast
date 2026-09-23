# ADR 0003: Train on archived weather forecasts, not observed weather

**Status:** Accepted
**Date:** 2026-09-21

## Context

Weather (especially temperature) is the strongest driver of electricity
demand after calendar effects. When the model runs in production to predict
tomorrow, only tomorrow's *forecast* weather exists.

## Decision

Use Open-Meteo's Historical Forecast API, which archives past forecasts as
they were issued, for training and evaluation. Observed-weather archives
(e.g. ERA5 reanalysis) are not used as model inputs.

Seven GB population centres are queried; each carries an approximate
population weight so Phase 2 can build a national demand-weighted
temperature.

## Consequences

- Offline accuracy honestly reflects production conditions; training on
  observed weather would leak information and overstate accuracy.
- The archive begins around 2022, which bounds our usable training history.
  Our demand history starts in 2023, so this is not currently a constraint.
- Point locations are an approximation of national weather; a gridded,
  demand-weighted average is a possible future improvement.
