import numpy as np
import pandas as pd
import pytest

from gridcast.features.build import FEATURE_COLUMNS, TARGET, build_features
from gridcast.models import MODEL_NAMES, build_model
from gridcast.models.gbm import QuantileGBM
from tests.conftest import make_demand, make_weather


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features(make_demand(days=120), make_weather(days=121))


def test_unknown_model_names_are_rejected_helpfully() -> None:
    with pytest.raises(KeyError, match="Available"):
        build_model("random_forest_of_dreams")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_sensible_numbers(name: str, features: pd.DataFrame) -> None:
    train, test = features.iloc[:-480], features.iloc[-480:]
    model = build_model(name)
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    predicted = model.predict(test[FEATURE_COLUMNS])

    assert len(predicted) == len(test)
    assert np.isfinite(predicted).all()
    assert (predicted > 5_000).all() and (predicted < 70_000).all()


def test_baseline_copies_the_lag_column(features: pd.DataFrame) -> None:
    model = build_model("naive_last_week")
    model.fit(features[FEATURE_COLUMNS], features[TARGET])
    predicted = model.predict(features[FEATURE_COLUMNS])
    assert np.allclose(predicted, features["nd_lag_7d_mw"].to_numpy())


def test_seasonal_profile_learns_weekday_and_weekend_shapes(features: pd.DataFrame) -> None:
    model = build_model("seasonal_profile")
    model.fit(features[FEATURE_COLUMNS], features[TARGET])
    predicted = model.predict(features[FEATURE_COLUMNS])
    weekday = predicted[features["is_weekend"].to_numpy() == 0].mean()
    weekend = predicted[features["is_weekend"].to_numpy() == 1].mean()
    assert weekday > weekend  # synthetic data has lower weekend demand


def test_learned_models_beat_the_naive_baseline(features: pd.DataFrame) -> None:
    """A model that cannot beat 'same time last week' is not worth deploying."""
    train, test = features.iloc[:-480], features.iloc[-480:]
    errors = {}
    for name in ("naive_last_week", "ridge", "lightgbm"):
        model = build_model(name)
        model.fit(train[FEATURE_COLUMNS], train[TARGET])
        predicted = model.predict(test[FEATURE_COLUMNS])
        errors[name] = float(np.mean(np.abs(predicted - test[TARGET].to_numpy())))

    assert errors["lightgbm"] < errors["naive_last_week"]
    assert errors["ridge"] < errors["naive_last_week"]


def test_models_that_can_explain_themselves_do(features: pd.DataFrame) -> None:
    for name in ("ridge", "lightgbm"):
        model = build_model(name)
        model.fit(features[FEATURE_COLUMNS], features[TARGET])
        importance = model.feature_importance()
        assert importance is not None
        assert set(importance.index) == set(FEATURE_COLUMNS)


def test_quantile_forecasts_are_ordered_and_contain_the_median(
    features: pd.DataFrame,
) -> None:
    train, test = features.iloc[:-480], features.iloc[-480:]
    model = QuantileGBM(n_estimators=60)
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    quantiles = model.predict_quantiles(test[FEATURE_COLUMNS])

    assert list(quantiles.columns) == ["p10", "p50", "p90"]
    assert (quantiles["p10"] <= quantiles["p50"]).all()
    assert (quantiles["p50"] <= quantiles["p90"]).all()
    assert np.allclose(model.predict(test[FEATURE_COLUMNS]), quantiles["p50"].to_numpy())
