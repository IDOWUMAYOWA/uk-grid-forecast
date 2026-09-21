"""Data contracts: the rules our cleaned data must always satisfy.

A contract is written once and checked on every run. If NESO changes a file
format or sends bad values, we find out immediately with a clear error,
instead of silently training a model on broken data.

Two kinds of failure are handled differently:

- Structural problems (a required column missing, wrong types) stop the run.
- A small number of bad rows (e.g. an impossible demand value) are moved to
  a quarantine file for inspection, and the run continues. If too many rows
  are bad, we stop anyway, because that signals a real problem upstream.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from gridcast.errors import DataQualityError
from gridcast.logging import get_logger
from gridcast.settlement import periods_in_day

log = get_logger(__name__)

# GB national demand has never been below ~10 GW or above ~62 GW. The bounds
# are deliberately loose: they catch nonsense (zeros, unit errors), not
# unusual-but-real days.
_DEMAND_MW = pa.Check.in_range(5_000, 70_000)


def _period_fits_day(df: pd.DataFrame) -> pd.Series:
    """Settlement period must exist on that date (46/48/50-period days)."""
    return df["settlement_period"] <= periods_in_day(df["settlement_date"])


DEMAND_SCHEMA = pa.DataFrameSchema(
    columns={
        "settlement_date": pa.Column("datetime64[ns]"),
        "settlement_period": pa.Column("int64", pa.Check.in_range(1, 50)),
        "timestamp_utc": pa.Column(pd.DatetimeTZDtype(unit="ns", tz="UTC"), unique=True),
        "nd_mw": pa.Column("float64", _DEMAND_MW),
        "tsd_mw": pa.Column("float64", pa.Check.in_range(5_000, 80_000)),
    },
    checks=[pa.Check(_period_fits_day, error="settlement_period beyond end of day")],
    coerce=True,
    strict=False,  # optional columns (interconnectors etc.) are allowed
)


def validate_with_quarantine(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    max_invalid_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate rows, returning (valid_rows, quarantined_rows).

    Raises DataQualityError on structural failures or too many bad rows.
    """
    df = df.reset_index(drop=True)
    try:
        return schema.validate(df, lazy=True), df.iloc[0:0]
    except SchemaErrors as err:
        failures = err.failure_cases

    if failures["index"].isna().any():
        structural = failures[failures["index"].isna()]
        raise DataQualityError(
            f"Structural contract failure: {structural[['column', 'check']].to_dict('records')}"
        )

    bad_rows = pd.Index(failures["index"].astype("int64").unique())
    fraction = len(bad_rows) / len(df)
    log.warning(
        "contract.rows_quarantined",
        rows=len(bad_rows),
        fraction=round(fraction, 5),
        checks=sorted(failures["check"].astype(str).unique().tolist()),
    )
    if fraction > max_invalid_fraction:
        raise DataQualityError(
            f"{fraction:.2%} of rows failed validation "
            f"(limit {max_invalid_fraction:.2%}); refusing to continue."
        )

    quarantined = df.loc[bad_rows]
    valid = schema.validate(df.drop(index=bad_rows))
    return valid, quarantined
