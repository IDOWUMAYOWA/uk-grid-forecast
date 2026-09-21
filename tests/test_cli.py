from typer.testing import CliRunner

from gridcast import __version__
from gridcast.cli import app

runner = CliRunner()


def test_info_command() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
