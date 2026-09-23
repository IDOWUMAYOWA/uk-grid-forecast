"""Command-line entry point: ``gridcast <command>``.

Every pipeline stage (ingest, build-features, train, predict, monitor) is
exposed here so the same code runs locally, in CI and in the scheduler.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Annotated

import httpx
import pandas as pd
import typer

from gridcast import __version__, data
from gridcast.config import Settings, get_settings
from gridcast.errors import GridcastError
from gridcast.features.build import build_and_store, features_path
from gridcast.http import make_client
from gridcast.ingest import elexon_generation, neso_demand, weather
from gridcast.logging import configure_logging, get_logger
from gridcast.models import MODEL_NAMES
from gridcast.models.registry import BASELINE_NAMES
from gridcast.train import compare_models, quantile_report, train

app = typer.Typer(help="GB electricity demand forecasting pipeline.", no_args_is_help=True)
ingest_app = typer.Typer(help="Download, validate and store source data.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")

log = get_logger(__name__)

StartYear = Annotated[int, typer.Option(help="First year to ingest.")]
EndYear = Annotated[int | None, typer.Option(help="Last year (default: this year).")]
Force = Annotated[bool, typer.Option(help="Re-download files we already have.")]


@app.callback()
def main() -> None:
    configure_logging(get_settings())


@contextmanager
def _client(source: str) -> Iterator[httpx.Client]:
    """HTTP client for a CLI command; turns known errors into exit code 1."""
    try:
        with make_client(get_settings()) as client:
            yield client
    except GridcastError as exc:
        log.error("ingest.failed", source=source, error=str(exc))
        raise typer.Exit(code=1) from exc


def _this_year() -> int:
    return datetime.now(UTC).year


@app.command()
def info() -> None:
    """Show version and resolved configuration."""
    settings = get_settings()
    log.info(
        "gridcast.info", version=__version__, env=settings.env, data_dir=str(settings.data_dir)
    )
    typer.echo(f"gridcast {__version__} (env={settings.env})")


@ingest_app.command("neso-demand")
def ingest_neso_demand(
    start_year: StartYear = 2023, end_year: EndYear = None, force: Force = False
) -> None:
    """NESO national demand history (yearly files, to end of last month)."""
    with _client("neso_demand") as client:
        reports = neso_demand.ingest(
            client, get_settings(), start_year, end_year or _this_year(), force=force
        )
    typer.echo(f"\n{'year':<9}{'rows':>8}{'quarantined':>13}{'downloaded':>12}")
    for r in reports:
        typer.echo(
            f"{r.year:<9}{r.rows:>8}{r.quarantined:>13}{'yes' if r.downloaded else 'no':>12}"
        )


@ingest_app.command("neso-demand-update")
def ingest_neso_demand_update() -> None:
    """NESO recent demand (first day of last month up to today)."""
    with _client("neso_demand_update") as client:
        rows, bad = neso_demand.ingest_update(client, get_settings())
    typer.echo(f"\nneso-demand-update: {rows} rows, {bad} quarantined")


@ingest_app.command("elexon-generation")
def ingest_elexon_generation(
    start_year: StartYear = 2023, end_year: EndYear = None, force: Force = False
) -> None:
    """Elexon generation by fuel type (supply side), month by month."""
    end = date(end_year, 12, 31) if end_year else datetime.now(UTC).date()
    with _client("elexon_generation") as client:
        reports = elexon_generation.ingest(
            client, get_settings(), date(start_year, 1, 1), end, force=force
        )
    typer.echo(f"\n{'month':<9}{'rows':>8}{'quarantined':>13}{'downloaded':>12}")
    for r in reports:
        typer.echo(
            f"{r.month:<9}{r.rows:>8}{r.quarantined:>13}{'yes' if r.downloaded else 'no':>12}"
        )


@ingest_app.command("weather")
def ingest_weather(
    start_year: StartYear = 2023, end_year: EndYear = None, force: Force = False
) -> None:
    """Open-Meteo historical weather forecasts for GB demand centres."""
    with _client("weather") as client:
        reports = weather.ingest(
            client, get_settings(), start_year, end_year or _this_year(), force=force
        )
    typer.echo(f"\n{'location':<12}{'year':<6}{'rows':>7}{'quarantined':>13}{'downloaded':>12}")
    for r in reports:
        typer.echo(
            f"{r.location:<12}{r.year:<6}{r.rows:>7}{r.quarantined:>13}"
            f"{'yes' if r.downloaded else 'no':>12}"
        )


@ingest_app.command("all")
def ingest_all(start_year: StartYear = 2023) -> None:
    """Run every ingestion step, in order. This is what the scheduler runs."""
    steps: list[tuple[str, Callable[[], None]]] = [
        ("neso-demand", lambda: ingest_neso_demand(start_year=start_year)),
        ("neso-demand-update", ingest_neso_demand_update),
        ("elexon-generation", lambda: ingest_elexon_generation(start_year=start_year)),
        ("weather", lambda: ingest_weather(start_year=start_year)),
    ]
    for name, step in steps:
        typer.echo(f"\n=== {name} ===")
        step()


@app.command("build-features")
def build_features_command() -> None:
    """Turn ingested data into the model-ready feature table."""
    settings = get_settings()
    try:
        features = build_and_store(settings)
    except (GridcastError, FileNotFoundError) as exc:
        log.error("features.failed", error=str(exc))
        raise typer.Exit(code=1) from exc

    first, last = features["timestamp_utc"].min(), features["timestamp_utc"].max()
    typer.echo(
        f"\n{len(features):,} rows x {len(features.columns)} columns "
        f"({first:%Y-%m-%d} to {last:%Y-%m-%d})\nsaved to {features_path(settings)}"
    )


Folds = Annotated[int, typer.Option(help="Number of walk-forward folds.")]
TestDays = Annotated[int, typer.Option(help="Days of held-out data per fold.")]


@app.command("train")
def train_command(
    model: Annotated[str, typer.Option(help=f"One of: {', '.join(MODEL_NAMES)}")] = "lightgbm",
    folds: Folds = 5,
    test_days: TestDays = 28,
    track: Annotated[bool, typer.Option(help="Record the run in MLflow.")] = True,
) -> None:
    """Evaluate a model with walk-forward validation, then fit it on all data."""
    try:
        result = train(get_settings(), model, n_folds=folds, test_days=test_days, track=track)
    except (GridcastError, KeyError, ValueError) as exc:
        log.error("train.failed", model=model, error=str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(f"\n{model}: walk-forward accuracy over {folds} folds")
    for name, value in result.overall.items():
        typer.echo(f"  {name:<14}{value:>12,.1f}")
    typer.echo("\nBy segment:")
    typer.echo(
        result.by_segment[["dimension", "segment", "mae_mw", "mape_pct"]].to_string(index=False)
    )


@app.command("compare")
def compare_command(folds: Folds = 5, test_days: TestDays = 28) -> None:
    """Score baselines and models on identical folds, best first."""
    names = [*BASELINE_NAMES, "ridge", "lightgbm"]
    try:
        table = compare_models(get_settings(), names, n_folds=folds, test_days=test_days)
    except (GridcastError, ValueError) as exc:
        log.error("compare.failed", error=str(exc))
        raise typer.Exit(code=1) from exc

    best = table.iloc[0]
    baseline = table[table.model == "naive_last_week"].iloc[0]
    improvement = (1 - best.mae_mw / baseline.mae_mw) * 100
    typer.echo("\n" + table.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    typer.echo(f"\nBest: {best.model} - {improvement:.0f}% lower MAE than copying last week")


@app.command("quantiles")
def quantiles_command(folds: Folds = 3, test_days: TestDays = 28) -> None:
    """Check that P10-P90 forecast intervals are honest."""
    try:
        report = quantile_report(get_settings(), n_folds=folds, test_days=test_days)
    except (GridcastError, ValueError) as exc:
        log.error("quantiles.failed", error=str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo("\n" + report.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    typer.echo("\nCoverage should be close to 0.80 for a P10-P90 interval.")


@app.command()
def status() -> None:
    """Show how much data we have for each source, and up to when."""
    settings = get_settings()
    loaders: list[tuple[str, Callable[[Settings], pd.DataFrame]]] = [
        ("demand (NESO)", data.load_demand),
        ("generation (Elexon)", data.load_generation),
        ("weather (Open-Meteo)", data.load_weather),
    ]
    typer.echo(f"{'dataset':<22}{'rows':>10}  {'first (UTC)':<18}{'last (UTC)':<18}")
    for name, loader in loaders:
        try:
            df = loader(settings)
        except FileNotFoundError:
            typer.echo(f"{name:<22}{'-':>10}  not ingested yet")
            continue
        ts = df["timestamp_utc"]
        typer.echo(
            f"{name:<22}{len(df):>10,}  {ts.min():%Y-%m-%d %H:%M}  {ts.max():%Y-%m-%d %H:%M}"
        )


if __name__ == "__main__":
    app()
