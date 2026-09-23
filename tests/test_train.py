import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from gridcast.config import Settings
from gridcast.errors import GridcastError
from gridcast.features.build import features_path
from gridcast.storage import write_parquet_atomic
from gridcast.train import (
    compare_models,
    evaluate_model,
    load_features,
    metadata_path,
    model_path,
    train,
)
from tests.conftest import make_demand, make_weather


@pytest.fixture(scope="module")
def feature_table() -> pd.DataFrame:
    from gridcast.features.build import build_features

    return build_features(make_demand(days=300), make_weather(days=301))


@pytest.fixture
def settings(tmp_path: Path, feature_table: pd.DataFrame) -> Settings:
    s = Settings(data_dir=tmp_path, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    write_parquet_atomic(feature_table, features_path(s))
    return s


def test_missing_feature_table_gives_a_useful_message(tmp_path: Path) -> None:
    with pytest.raises(GridcastError, match="build-features"):
        load_features(Settings(data_dir=tmp_path))


def test_evaluation_scores_every_fold_and_predicts_every_test_row(
    feature_table: pd.DataFrame,
) -> None:
    result = evaluate_model(feature_table, "naive_last_week", n_folds=3, test_days=14)

    assert len(result.fold_metrics) == 3
    assert result.predictions["fold"].nunique() == 3
    assert result.overall["mae_mw"] > 0
    assert not result.by_segment.empty
    # Backtest predictions cover distinct days: folds must not overlap.
    assert result.predictions["timestamp_utc"].is_unique


def test_training_saves_a_usable_model_and_metadata(settings: Settings) -> None:
    result = train(settings, "ridge", n_folds=2, test_days=14, track=False)

    path = model_path(settings, "ridge")
    assert path.exists()
    metadata = json.loads(metadata_path(settings, "ridge").read_text())
    assert metadata["model_name"] == "ridge"
    assert metadata["training_rows"] > 0
    assert metadata["walk_forward_metrics"]["mae_mw"] == pytest.approx(result.overall["mae_mw"])

    # The saved file must be a working model, not just bytes on disk.
    restored = joblib.load(path)
    features = load_features(settings)
    from gridcast.features.build import FEATURE_COLUMNS

    assert len(restored.predict(features[FEATURE_COLUMNS].head(48))) == 48


def test_backtest_predictions_are_stored_for_later_analysis(settings: Settings) -> None:
    train(settings, "naive_last_week", n_folds=2, test_days=14, track=False)
    stored = pd.read_parquet(settings.data_dir / "evaluation" / "naive_last_week_backtest.parquet")
    assert {"actual_mw", "predicted_mw", "fold"} <= set(stored.columns)


@pytest.mark.parametrize("model_name", ["naive_two_days", "ridge"])
def test_mlflow_records_the_run(settings: Settings, model_name: str) -> None:
    """Covers both a model without feature importance and one with it."""
    import mlflow

    train(settings, model_name, n_folds=2, test_days=14, track=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=["gridcast-demand"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.model"] == model_name
    assert runs.iloc[0]["metrics.mae_mw"] > 0


def test_comparison_ranks_models_by_error(settings: Settings) -> None:
    table = compare_models(settings, ["naive_last_week", "lightgbm"], n_folds=2, test_days=14)
    assert next(iter(table.columns)) == "model"
    assert table["mae_mw"].is_monotonic_increasing  # best first
    assert table.iloc[0]["model"] == "lightgbm"
