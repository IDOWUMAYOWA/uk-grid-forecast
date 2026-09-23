# ADR 0005: Walk-forward validation and model selection

**Status:** Accepted
**Date:** 2026-09-23

## Context

Accuracy claims are only meaningful if the evaluation resembles production.
Random k-fold splitting trains on the future to predict the past and badly
overstates accuracy for time series.

## Decision

1. **Walk-forward (rolling origin) validation** with an expanding training
   window: train up to a date, leave a one-day gap, score the next 28 days,
   then move the origin forward. Five folds by default.
2. **The gap is mandatory.** A forecast issued on the morning of day D cannot
   have been trained on D's outturn. A test asserts every fold's training
   data ends at least two days before its test window starts.
3. **Baselines are first-class models**, evaluated on identical folds:
   copying last week, copying two days ago, and a learned weekday/weekend
   profile. Reported accuracy is always stated relative to them.
4. **MAE in MW is the headline metric** because that is what an operator
   acts on; RMSE, MAPE, bias and worst-case error are reported alongside.
   Accuracy is also broken down by season, day type and evening peak, since
   aggregates hide the failures that matter.
5. **LightGBM optimises MAE** (`regression_l1`) rather than squared error,
   so a handful of extreme days cannot dominate the fit.
6. **Quantile models (P10/P50/P90)** provide intervals, scored with pinball
   loss and empirical coverage. A P10-P90 interval should contain about 80%
   of outturns; systematic deviation means the uncertainty is mis-stated.
7. **Every run is tracked in MLflow** (local SQLite by default, a hosted
   server via `GRIDCAST_MLFLOW_TRACKING_URI`), including parameters, per-fold
   metrics, segment breakdowns, feature importance and the model artifact.
8. **The deployed model is refit on all history** after evaluation; its
   reported accuracy remains the walk-forward estimate, never a score on
   data it trained on.

## Consequences

- Reported accuracy will look worse than published figures that use random
  splits or same-day lags, and will be closer to what production delivers.
- Refitting on all data means the deployed model is never scored directly;
  Phase 7 monitoring closes this gap by comparing live forecasts with
  outturn as it arrives.
- Comparison against NESO's own published day-ahead forecast is the next
  step and needs its own ingestion of that dataset.
