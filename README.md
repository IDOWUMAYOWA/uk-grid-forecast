# gridcast ⚡ — GB Electricity Demand & Supply-Margin Forecasting

[![CI](https://github.com/IDOWUMAYOWA/uk-grid-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/IDOWUMAYOWA/uk-grid-forecast/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)
![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-2a6db2)

An end-to-end, production-style ML system that forecasts **day-ahead half-hourly
national electricity demand for Great Britain**, forecasts weather-driven
renewable generation, and reports the **expected supply margin** — benchmarked
against the system operator's (NESO) own published forecast.

## Why this project

GB's system operator must balance supply and demand every half hour. As wind and
solar grow, the hard problem shifts from forecasting demand to forecasting
**net demand** (demand minus weather-driven generation) — the residual that
dispatchable plant and interconnectors must cover. This project tackles both,
with the engineering practices of a production ML service.

## Architecture

```
 NESO API ─┐                                     ┌─► FastAPI  /forecast
 Elexon API ├─► Ingest ─► Validate ─► Features ─► Train ─► Registry ─┤
 Open-Meteo ┘   (raw     (pandera   (parquet)   (LightGBM  (MLflow)   └─► Batch predictions ─► Dashboard
                 parquet)  contracts)             + quantiles)                     │
                                                                                   ▼
                                                              Monitoring: accuracy vs actuals,
                                                              data & prediction drift, alerts
```

## Roadmap

| Phase | Scope | Status |
|------:|-------|--------|
| 0 | Repo, Codespaces, CI, quality gates | ✅ |
| 1 | Data ingestion (NESO, Elexon, Open-Meteo) + data contracts | ⏳ |
| 2 | EDA + feature engineering | ⏳ |
| 3 | Baselines, LightGBM, walk-forward CV, MLflow tracking | ⏳ |
| 4 | Supply side: wind/solar forecasts, net demand, margin | ⏳ |
| 5 | Serving: FastAPI + Docker | ⏳ |
| 6 | Orchestration: scheduled daily pipeline | ⏳ |
| 7 | Monitoring: accuracy tracking, drift, alerting | ⏳ |
| 8 | Cloud deployment + public dashboard | ⏳ |

## Quickstart

**Codespaces (recommended):** click **Code → Codespaces → Create codespace on main**.
Everything installs automatically. Then:

```bash
make check          # lint + type-check + tests
uv run gridcast info
```

**Local:** install [uv](https://docs.astral.sh/uv/), then `make install && make check`.

## Project layout

```
src/gridcast/      application package (importable, typed, tested)
tests/             unit tests; integration tests marked @pytest.mark.integration
docs/adr/          architecture decision records
data/              local data lake (git-ignored, fully reproducible)
.devcontainer/     Codespaces / dev container definition
.github/           CI workflows, Dependabot, PR template
```

## Data sources

All public and free: [NESO Data Portal](https://www.neso.energy/data-portal)
(demand outturn, day-ahead forecasts), [Elexon Insights](https://developer.data.elexon.co.uk/)
(generation by fuel type, interconnectors), [Open-Meteo](https://open-meteo.com/)
(historical weather *forecasts*, to avoid look-ahead leakage).

## License

MIT
