from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from gridcast.evaluate import (
    interval_coverage,
    metrics,
    metrics_by_segment,
    pinball_loss,
    walk_forward_splits,
)


def dates(start: str = "2024-01-01", days: int = 400) -> pd.Series:
    return pd.Series(pd.date_range(start, periods=days * 48, freq="30min")).dt.normalize()


def test_folds_are_chronological_and_do_not_overlap() -> None:
    d = dates()
    folds = walk_forward_splits(d, n_folds=4, test_days=28)
    assert len(folds) == 4
    for earlier, later in pairwise(folds):
        assert earlier.test_end < later.test_start


def test_training_data_always_ends_before_the_test_window() -> None:
    """The core guarantee: no fold may train on its own test period."""
    d = dates()
    for fold in walk_forward_splits(d, n_folds=4, test_days=28, gap_days=1):
        train_days = d.iloc[fold.train_rows]
        test_days_ = d.iloc[fold.test_rows]
        assert train_days.max() < test_days_.min()
        # And a gap is left, because a forecast cannot use same-day outturn.
        assert (test_days_.min() - train_days.max()).days >= 2


def test_training_window_expands_with_each_fold() -> None:
    folds = walk_forward_splits(dates(), n_folds=4, test_days=28)
    sizes = [len(f.train_rows) for f in folds]
    assert sizes == sorted(sizes)


def test_too_little_history_is_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="Not enough history"):
        walk_forward_splits(dates(days=30), n_folds=5, test_days=28)


def test_metrics_are_computed_correctly() -> None:
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = pd.Series([110.0, 190.0, 300.0])
    result = metrics(actual, predicted)
    assert result["mae_mw"] == pytest.approx(20 / 3)
    assert result["bias_mw"] == pytest.approx(0.0)
    assert result["max_error_mw"] == pytest.approx(10.0)
    assert result["mape_pct"] == pytest.approx((0.10 + 0.05 + 0) / 3 * 100)


def test_perfect_forecast_scores_zero() -> None:
    actual = pd.Series([100.0, 200.0])
    assert metrics(actual, actual)["mae_mw"] == 0.0


def test_segment_breakdown_covers_seasons_days_and_peak() -> None:
    n = 96
    frame = pd.DataFrame(
        {
            "settlement_date": pd.date_range("2025-01-04", periods=n, freq="D"),
            "settlement_period": np.tile(np.arange(1, 49), 2),
            "actual_mw": np.full(n, 30_000.0),
            "predicted_mw": np.full(n, 31_000.0),
        }
    )
    table = metrics_by_segment(frame)
    assert set(table["dimension"]) == {"season", "day_type", "part_of_day"}
    assert {"weekday", "weekend"} <= set(table["segment"])
    assert (table["mae_mw"] == 1_000.0).all()


def test_pinball_loss_penalises_the_right_side() -> None:
    actual = pd.Series([100.0])
    # For the 10th percentile, over-forecasting should hurt more than under.
    assert pinball_loss(actual, pd.Series([120.0]), 0.1) > pinball_loss(
        actual, pd.Series([80.0]), 0.1
    )


def test_interval_coverage_counts_actuals_inside_the_band() -> None:
    actual = pd.Series([10.0, 20.0, 30.0, 40.0])
    lower = pd.Series([0.0, 0.0, 0.0, 0.0])
    upper = pd.Series([15.0, 25.0, 25.0, 35.0])
    assert interval_coverage(actual, lower, upper) == 0.5
