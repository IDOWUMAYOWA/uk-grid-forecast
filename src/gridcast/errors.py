"""Project-wide exception types.

Using our own exceptions lets callers (the CLI, the scheduler) tell the
difference between "the source is down" and "the data is bad".
"""


class GridcastError(Exception):
    """Base class for all gridcast errors."""


class SourceError(GridcastError):
    """An external data source returned something we could not use."""


class DataQualityError(GridcastError):
    """Data failed validation badly enough that we must stop."""
