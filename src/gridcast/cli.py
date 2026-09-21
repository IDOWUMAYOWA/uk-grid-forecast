"""Command-line entry point: ``gridcast <command>``.

Every pipeline stage (ingest, build-features, train, predict, monitor) will
be exposed here so the same code runs locally, in CI and in the scheduler.
"""

import typer

from gridcast import __version__
from gridcast.config import get_settings
from gridcast.logging import configure_logging, get_logger

app = typer.Typer(help="GB electricity demand forecasting pipeline.", no_args_is_help=True)


@app.callback()
def main() -> None:
    configure_logging(get_settings())


@app.command()
def info() -> None:
    """Show version and resolved configuration."""
    settings = get_settings()
    log = get_logger(__name__)
    log.info(
        "gridcast.info",
        version=__version__,
        env=settings.env,
        data_dir=str(settings.data_dir),
    )
    typer.echo(f"gridcast {__version__} (env={settings.env})")


if __name__ == "__main__":
    app()
