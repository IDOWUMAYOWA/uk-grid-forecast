"""Time-aware validation and forecast accuracy metrics.

Random train/test splits are wrong for time series: they let a model learn
from the future to predict the past. We use **walk-forward (rolling origin)**
validation instead, which mimics real life: train on everything up to a
point, forecast the next few weeks, move forward, repeat.

A one-day gap sits between training and test data, because a forecast issued
on the morning of day D cannot be trained on day D's outturn.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Evening peak (local): roughly 16:00-21:00, when the system is tightest and
# forecast errors are most expensive.
PEAK_PERIODS = range(33, 43)


@dataclass(frozen=True)
class Fold:
    """One train/test split in a walk-forward evaluation."""

    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: np.ndarray
    test_rows: np.ndarray


def walk_forward_splits(
    dates: pd.Series,
    n_folds: int = 5,
    test_days: int = 28,
    gap_days: int = 1,
    min_train_days: int = 180,
) -> list[Fold]:
    """Build expanding-window folds, oldest first.

    Each fold trains on all data up to `train_end`, skips `gap_days`, then
    tests on the following `test_days`.
    """
    dates = pd.to_datetime(dates).dt.normalize()
    last_day = dates.max()
    first_day = dates.min()

    folds: list[Fold] = []
    for i in reversed(range(n_folds)):
        test_end = last_day - pd.Timedelta(days=i * test_days)
        test_start = test_end - pd.Timedelta(days=test_days - 1)
        train_end = test_start - pd.Timedelta(days=gap_days + 1)

        if (train_end - first_day).days < min_train_days:
            continue

        train_rows = np.flatnonzero(dates <= train_end)
        test_rows = np.flatnonzero((dates >= test_start) & (dates <= test_end))
        if len(train_rows) == 0 or len(test_rows) == 0:
            continue

        folds.append(Fold(len(folds), train_end, test_start, test_end, train_rows, test_rows))

    if not folds:
        raise ValueError(
            f"Not enough history for {n_folds} folds of {test_days} days "
            f"with {min_train_days} days of training data."
        )
    return folds


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Headline accuracy metrics.

    MAE is in megawatts, which is what an operator cares about; MAPE is the
    percentage version that allows comparison across systems; bias shows
    whether we systematically over- or under-forecast.
    """
    error = predicted.to_numpy() - actual.to_numpy()
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error / actual.to_numpy())) * 100),
        "bias_mw": float(np.mean(error)),
        "max_error_mw": float(np.max(np.abs(error))),
    }


def metrics_by_segment(predictions: pd.DataFrame) -> pd.DataFrame:
    """Break accuracy down by season, day type and peak hours.

    An aggregate number hides the failures that matter: a model can look fine
    overall while being poor on winter peaks or bank holidays.
    """
    df = predictions.copy()
    df["season"] = df["settlement_date"].dt.month.map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )
    df["day_type"] = np.where(df["settlement_date"].dt.dayofweek >= 5, "weekend", "weekday")
    df["part_of_day"] = np.where(
        df["settlement_period"].isin(PEAK_PERIODS), "evening peak", "off peak"
    )

    rows = []
    for dimension in ("season", "day_type", "part_of_day"):
        for value, group in df.groupby(dimension, observed=True):
            rows.append(
                {"dimension": dimension, "segment": str(value), "rows": len(group)}
                | metrics(group["actual_mw"], group["predicted_mw"])
            )
    return pd.DataFrame(rows)


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    """Scoring rule for quantile forecasts (lower is better)."""
    error = actual.to_numpy() - predicted.to_numpy()
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Share of actual values that fell inside the prediction interval."""
    inside = (actual.to_numpy() >= lower.to_numpy()) & (actual.to_numpy() <= upper.to_numpy())
    return float(np.mean(inside))
