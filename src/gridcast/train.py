"""Train, evaluate and record models.

Two things happen here, and keeping them separate matters:

1. **Evaluation** runs walk-forward validation to estimate how accurate the
   model will be on days it has never seen. This is the number we report.
2. **Training the deliverable** then refits the same configuration on *all*
   available history, because the model we deploy should know as much as
   possible. Its accuracy is the one measured in step 1, not on its own
   training data.

Every run is recorded in MLflow: parameters, metrics per fold, accuracy by
season and day type, feature importance, and the fitted model itself. Runs
are comparable, and a result you cannot reproduce is not a result.
"""

import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from gridcast import __version__
from gridcast.config import Settings
from gridcast.errors import GridcastError
from gridcast.evaluate import Fold, metrics, metrics_by_segment, walk_forward_splits
from gridcast.features.build import FEATURE_COLUMNS, TARGET, features_path
from gridcast.logging import get_logger
from gridcast.models import build_model
from gridcast.models.gbm import QuantileGBM
from gridcast.storage import write_parquet_atomic

log = get_logger(__name__)

EXPERIMENT = "gridcast-demand"


@dataclass
class EvaluationResult:
    model_name: str
    fold_metrics: pd.DataFrame
    overall: dict[str, float]
    predictions: pd.DataFrame
    by_segment: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_features(settings: Settings) -> pd.DataFrame:
    path = features_path(settings)
    if not path.exists():
        raise GridcastError(f"No feature table at {path}. Run `gridcast build-features` first.")
    return pd.read_parquet(path)


def evaluate_model(
    features: pd.DataFrame,
    model_name: str,
    n_folds: int = 5,
    test_days: int = 28,
) -> EvaluationResult:
    """Walk-forward evaluation: train on the past, score the following weeks."""
    folds = walk_forward_splits(features["settlement_date"], n_folds=n_folds, test_days=test_days)

    fold_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for fold in folds:
        train = features.iloc[fold.train_rows]
        test = features.iloc[fold.test_rows]

        model = build_model(model_name)
        model.fit(train[FEATURE_COLUMNS], train[TARGET])
        predicted = model.predict(test[FEATURE_COLUMNS])

        frame = pd.DataFrame(
            {
                "timestamp_utc": test["timestamp_utc"].to_numpy(),
                "settlement_date": test["settlement_date"].to_numpy(),
                "settlement_period": test["settlement_period"].to_numpy(),
                "actual_mw": test[TARGET].to_numpy(),
                "predicted_mw": predicted,
                "fold": fold.index,
            }
        )
        predictions.append(frame)

        fold_metrics = metrics(frame["actual_mw"], frame["predicted_mw"])
        fold_rows.append(
            {
                "fold": fold.index,
                "train_rows": len(train),
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **fold_metrics,
            }
        )
        log.info("evaluate.fold", model=model_name, fold=fold.index, **fold_metrics)

    all_predictions = pd.concat(predictions, ignore_index=True)
    return EvaluationResult(
        model_name=model_name,
        fold_metrics=pd.DataFrame(fold_rows),
        overall=metrics(all_predictions["actual_mw"], all_predictions["predicted_mw"]),
        predictions=all_predictions,
        by_segment=metrics_by_segment(all_predictions),
    )


def model_path(settings: Settings, model_name: str) -> Path:
    return settings.data_dir / "models" / f"{model_name}.joblib"


def metadata_path(settings: Settings, model_name: str) -> Path:
    return settings.data_dir / "models" / f"{model_name}.metadata.json"


def train_final_model(
    features: pd.DataFrame, model_name: str, settings: Settings, evaluation: EvaluationResult
) -> Path:
    """Refit on all history and save the model plus a description of it."""
    model = build_model(model_name)
    model.fit(features[FEATURE_COLUMNS], features[TARGET])

    path = model_path(settings, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    metadata = {
        "model_name": model_name,
        "gridcast_version": __version__,
        "python_version": platform.python_version(),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "training_rows": len(features),
        "training_start": str(features["settlement_date"].min().date()),
        "training_end": str(features["settlement_date"].max().date()),
        "features": FEATURE_COLUMNS,
        "walk_forward_metrics": evaluation.overall,
    }
    metadata_path(settings, model_name).write_text(json.dumps(metadata, indent=2))
    log.info("train.saved", model=model_name, path=str(path), rows=len(features))
    return path


def _log_to_mlflow(
    settings: Settings, evaluation: EvaluationResult, features: pd.DataFrame, model_file: Path
) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=evaluation.model_name):
        mlflow.log_params(
            {
                "model": evaluation.model_name,
                "folds": len(evaluation.fold_metrics),
                "n_features": len(FEATURE_COLUMNS),
                "training_rows": len(features),
                "training_end": str(features["settlement_date"].max().date()),
                "gridcast_version": __version__,
            }
        )
        mlflow.log_metrics(evaluation.overall)
        for row in evaluation.fold_metrics.to_dict("records"):
            for metric in ("mae_mw", "mape_pct", "rmse_mw"):
                mlflow.log_metric(f"fold_{metric}", float(row[metric]), step=int(row["fold"]))

        mlflow.log_table(evaluation.by_segment, "metrics_by_segment.json")
        model = joblib.load(model_file)
        importance = model.feature_importance()
        if importance is not None:
            # Build the frame explicitly: Series.reset_index keyword arguments
            # differ between pandas versions.
            table = pd.DataFrame(
                {"feature": list(importance.index), "importance": importance.to_numpy()}
            )
            mlflow.log_table(table, "feature_importance.json")
        mlflow.log_artifact(str(model_file))


def train(
    settings: Settings,
    model_name: str,
    n_folds: int = 5,
    test_days: int = 28,
    track: bool = True,
) -> EvaluationResult:
    """Evaluate a model, refit it on all data, save it, and record the run."""
    features = load_features(settings)
    evaluation = evaluate_model(features, model_name, n_folds=n_folds, test_days=test_days)
    model_file = train_final_model(features, model_name, settings, evaluation)

    write_parquet_atomic(
        evaluation.predictions,
        settings.data_dir / "evaluation" / f"{model_name}_backtest.parquet",
    )
    if track:
        _log_to_mlflow(settings, evaluation, features, model_file)
    return evaluation


def compare_models(
    settings: Settings, model_names: list[str], n_folds: int = 5, test_days: int = 28
) -> pd.DataFrame:
    """Evaluate several models on identical folds and rank them."""
    features = load_features(settings)
    rows = []
    for name in model_names:
        result = evaluate_model(features, name, n_folds=n_folds, test_days=test_days)
        rows.append({"model": name, **result.overall})
    return pd.DataFrame(rows).sort_values("mae_mw").reset_index(drop=True)


def quantile_report(settings: Settings, n_folds: int = 3, test_days: int = 28) -> pd.DataFrame:
    """How well do the P10-P90 intervals actually behave?"""
    from gridcast.evaluate import interval_coverage, pinball_loss

    features = load_features(settings)
    folds: list[Fold] = walk_forward_splits(
        features["settlement_date"], n_folds=n_folds, test_days=test_days
    )
    rows = []
    for fold in folds:
        train_df = features.iloc[fold.train_rows]
        test_df = features.iloc[fold.test_rows]
        model = QuantileGBM()
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET])
        quantiles = model.predict_quantiles(test_df[FEATURE_COLUMNS])
        actual = test_df[TARGET].reset_index(drop=True)
        quantiles = quantiles.reset_index(drop=True)
        rows.append(
            {
                "fold": fold.index,
                "coverage_p10_p90": interval_coverage(actual, quantiles["p10"], quantiles["p90"]),
                "pinball_p10": pinball_loss(actual, quantiles["p10"], 0.1),
                "pinball_p50": pinball_loss(actual, quantiles["p50"], 0.5),
                "pinball_p90": pinball_loss(actual, quantiles["p90"], 0.9),
                "mean_interval_width_mw": float((quantiles["p90"] - quantiles["p10"]).mean()),
            }
        )
    return pd.DataFrame(rows)
