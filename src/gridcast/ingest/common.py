"""Steps shared by every ingestion source."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from gridcast.config import Settings
from gridcast.contracts import validate_with_quarantine
from gridcast.storage import write_parquet_atomic


def validate_and_store(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    settings: Settings,
    processed: Path,
    quarantine: Path,
) -> tuple[int, int]:
    """Validate against a contract, then write valid and quarantined rows.

    Returns (valid_rows, quarantined_rows).
    """
    valid, bad = validate_with_quarantine(df, schema, settings.max_invalid_row_fraction)
    write_parquet_atomic(valid, processed)
    if len(bad):
        write_parquet_atomic(bad, quarantine)
    return len(valid), len(bad)


def is_recent(year: int, month: int, today: pd.Timestamp, months: int = 2) -> bool:
    """True if (year, month) is within the last `months` months, inclusive.

    Recent data is still being added to or revised at the source, so we
    always re-download it; older data is downloaded once and cached.
    """
    age = (today.year - year) * 12 + (today.month - month)
    return age < months
