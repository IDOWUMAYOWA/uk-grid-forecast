"""Command-line entry point: ``gridcast <command>``.

Every pipeline stage (ingest, build-features, train, predict, monitor) is
exposed here so the same code runs locally, in CI and in the scheduler.
"""

from datetime import UTC, datetime
from typing import Annotated

import typer

from gridcast import __version__
from gridcast.config import get_settings
from gridcast.errors import GridcastError
from gridcast.http import make_client
from gridcast.ingest import neso_demand
from gridcast.logging import configure_logging, get_logger

app = typer.Typer(help="GB electricity demand forecasting pipeline.", no_args_is_help=True)
ingest_app = typer.Typer(help="Download, validate and store source data.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")

log = get_logger(__name__)


@app.callback()
def main() -> None:
    configure_logging(get_settings())


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
    start_year: Annotated[int, typer.Option(help="First year to ingest.")] = 2017,
    end_year: Annotated[int | None, typer.Option(help="Last year (default: this year).")] = None,
    force: Annotated[bool, typer.Option(help="Re-download files we already have.")] = False,
) -> None:
    """Ingest NESO historic national demand (half-hourly, GB)."""
    settings = get_settings()
    end = end_year or datetime.now(UTC).year
    try:
        with make_client(settings) as client:
            reports = neso_demand.ingest(client, settings, start_year, end, force=force)
    except GridcastError as exc:
        log.error("ingest.failed", source="neso_demand", error=str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(f"\n{'year':<6}{'rows':>8}{'quarantined':>13}{'downloaded':>12}")
    for r in reports:
        typer.echo(
            f"{r.year:<6}{r.rows:>8}{r.quarantined:>13}{'yes' if r.downloaded else 'no':>12}"
        )


if __name__ == "__main__":
    app()
