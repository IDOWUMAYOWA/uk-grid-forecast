import pytest

from gridcast.features.lags import SAME_PERIOD_LAG_DAYS, add_lag_features
from tests.conftest import make_demand


def test_lag_matches_the_same_period_days_earlier() -> None:
    df = add_lag_features(make_demand(days=20))
    row = df[(df["settlement_date"] == "2025-01-15") & (df["settlement_period"] == 20)].iloc[0]
    source = df[(df["settlement_date"] == "2025-01-13") & (df["settlement_period"] == 20)].iloc[0]
    assert row["nd_lag_2d_mw"] == pytest.approx(source["nd_mw"])


def test_previous_day_summaries_come_from_two_days_before() -> None:
    df = add_lag_features(make_demand(days=20))
    target_day = df[df["settlement_date"] == "2025-01-15"]
    source_day = df[df["settlement_date"] == "2025-01-13"]["nd_mw"]
    assert target_day["nd_prev_day_mean_mw"].iloc[0] == pytest.approx(source_day.mean())
    assert target_day["nd_prev_day_max_mw"].iloc[0] == pytest.approx(source_day.max())


def test_earliest_rows_have_no_history() -> None:
    df = add_lag_features(make_demand(days=20))
    first_day = df[df["settlement_date"] == "2025-01-01"]
    assert first_day[[f"nd_lag_{d}d_mw" for d in SAME_PERIOD_LAG_DAYS]].isna().all().all()


def test_no_lag_is_shorter_than_two_days() -> None:
    """Guards the rule that keeps day-ahead forecasting honest."""
    assert min(SAME_PERIOD_LAG_DAYS) >= 2


def test_lags_align_by_settlement_period_across_clock_change() -> None:
    # 26 October 2025 has 50 periods (clocks go back); 2 November has 48.
    df = add_lag_features(make_demand("2025-10-20", days=21))

    ordinary = df[(df["settlement_date"] == "2025-11-02") & (df["settlement_period"] == 40)]
    source = df[(df["settlement_date"] == "2025-10-26") & (df["settlement_period"] == 40)]
    assert ordinary["nd_lag_7d_mw"].iloc[0] == pytest.approx(source["nd_mw"].iloc[0])

    # Periods 49 and 50 exist only on the long day, so a week earlier there
    # is nothing to match: the lag is empty rather than silently wrong.
    long_day = df[(df["settlement_date"] == "2025-10-26") & (df["settlement_period"] > 48)]
    assert len(long_day) == 2
    assert long_day["nd_lag_7d_mw"].isna().all()
