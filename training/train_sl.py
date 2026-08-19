#!/usr/bin/env python3
"""Train the NeuroCompiler supervised immediate pass-quality model.

The model scores (program state, candidate LLVM pass) pairs. For the runtime
project, the default target is runtime_improvement_pct, not CompilerGym's
instruction-count step_reward.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from training.common import get_feature_cols, load_csv_rows, safe_float, split_by_column

LOGGER = logging.getLogger("train_sl")
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "processed" / "cbench_runtime_hybrid_dataset.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "supervised"
TARGETS = (
    "runtime_improvement_pct",
    "runtime_speedup",
    "runtime_reduction_sec",
    "step_reward",
    "z_runtime_improvement_pct",
)


def build_action_vocab(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    flags = sorted({r.get("pass_flag", "") or r.get("pass_name", "") for r in rows} - {""})
    if not flags:
        raise RuntimeError("No pass_flag/pass_name values found")
    return {flag: i for i, flag in enumerate(flags)}


def encode_row(
    row: Dict[str, str], feature_cols: Sequence[str], action_vocab: Dict[str, int]
) -> List[float]:
    """Encode state plus a one-hot candidate pass (no ordinal pass-ID bug)."""
    values = [safe_float(row.get(col, "")) or 0.0 for col in feature_cols]
    one_hot = [0.0] * len(action_vocab)
    flag = row.get("pass_flag", "") or row.get("pass_name", "")
    if flag not in action_vocab:
        raise ValueError(f"Unknown pass flag: {flag!r}")
    one_hot[action_vocab[flag]] = 1.0
    return values + one_hot


def target_value(row: Dict[str, str], target: str) -> float | None:
    value = safe_float(row.get(target, ""))
    if value is None or not math.isfinite(value):
        return None
    return value


def sample_weight_value(row: Dict[str, str], weight_col: Optional[str]) -> float:
    if not weight_col:
        return 1.0
    value = safe_float(row.get(weight_col, ""))
    if value is None:
        return 1.0
    return max(0.0, value)


def extract_xy(
    rows: Sequence[Dict[str, str]],
    feature_cols: Sequence[str],
    action_vocab: Dict[str, int],
    target: str,
    weight_col: Optional[str] = None,
) -> Tuple[List[List[float]], List[float], List[float], List[Dict[str, str]]]:
    X: List[List[float]] = []
    y: List[float] = []
    weights: List[float] = []
    kept: List[Dict[str, str]] = []
    for row in rows:
        value = target_value(row, target)
        flag = row.get("pass_flag", "") or row.get("pass_name", "")
        if value is None or flag not in action_vocab:
            continue
        X.append(encode_row(row, feature_cols, action_vocab))
        y.append(value)
        weights.append(sample_weight_value(row, weight_col))
        kept.append(row)
    return X, y, weights, kept


def get_model(model_type: str, seed: int):
    model_type = model_type.lower()
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=400, learning_rate=0.03, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, random_state=seed,
            )
        except ImportError:
            LOGGER.warning("LightGBM unavailable; using HistGradientBoosting")
    elif model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=400, learning_rate=0.03, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, random_state=seed,
                objective="reg:squarederror", n_jobs=-1,
            )
        except ImportError:
            LOGGER.warning("XGBoost unavailable; using HistGradientBoosting")
    elif model_type == "catboost":
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.03,
                random_seed=seed, verbose=False,
            )
        except ImportError:
            LOGGER.warning("CatBoost unavailable; using HistGradientBoosting")
    elif model_type in ("randomforest", "rf"):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=500, max_depth=14, min_samples_leaf=2,
            random_state=seed, n_jobs=-1,
        )
    elif model_type in ("mlp", "nn"):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            random_state=seed, verbose=False,
        )

    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=400, max_depth=8, learning_rate=0.04,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30, random_state=seed,
        verbose=0,
    )


def regression_metrics(model: Any, X: Sequence[Sequence[float]], y: Sequence[float]) -> Dict[str, float]:
    if not X:
        return {}
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    predictions = model.predict(X)
    mse = float(mean_squared_error(y, predictions))
    result = {
        "samples": len(y),
        "mae": float(mean_absolute_error(y, predictions)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": float(r2_score(y, predictions)) if len(y) >= 2 else float("nan"),
    }
    return result


def ranking_metrics(
    model: Any,
    rows: Sequence[Dict[str, str]],
    feature_cols: Sequence[str],
    action_vocab: Dict[str, int],
    target: str,
) -> Dict[str, float]:
    """Measure whether the model ranks the truly best pass highly per state."""
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if target_value(row, target) is not None:
            groups[(row.get("benchmark_uri", ""), row.get("pre_state_id", ""))].append(row)
    top1 = top3 = 0
    regrets: List[float] = []
    evaluated = 0
    for candidates in groups.values():
        if len(candidates) < 2:
            continue
        X = [encode_row(row, feature_cols, action_vocab) for row in candidates]
        predicted = list(map(float, model.predict(X)))
        actual = [float(target_value(row, target)) for row in candidates]  # type: ignore[arg-type]
        true_best = max(range(len(actual)), key=actual.__getitem__)
        ranking = sorted(range(len(predicted)), key=predicted.__getitem__, reverse=True)
        top1 += int(ranking[0] == true_best)
        top3 += int(true_best in ranking[:3])
        regrets.append(max(actual) - actual[ranking[0]])
        evaluated += 1
    if not evaluated:
        return {"groups": 0}
    return {
        "groups": evaluated,
        "top1_accuracy": top1 / evaluated,
        "top3_accuracy": top3 / evaluated,
        "mean_oracle_regret": sum(regrets) / len(regrets),
    }


def train(args: argparse.Namespace) -> Path:
    total_started = time.perf_counter()
    input_path = Path(args.input).expanduser().resolve()
    model_dir = Path(args.output_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = load_csv_rows(input_path, max_rows=args.max_rows)
    if args.target not in fieldnames:
        raise RuntimeError(
            f"Target {args.target!r} is not in {input_path}. Available runtime targets: "
            f"{[name for name in TARGETS if name in fieldnames]}"
        )
    splits = split_by_column(rows, split_col="dataset_split")
    feature_cols = get_feature_cols(fieldnames, use_norm=args.use_normalized)
    if not feature_cols:
        raise RuntimeError("No pre-state feature columns found")
    action_vocab = build_action_vocab(rows)

    weight_col = args.sample_weight_col
    if weight_col and weight_col not in fieldnames:
        raise RuntimeError(f"Sample weight column {weight_col!r} is not in {input_path}")

    X_train, y_train, train_weights, train_rows = extract_xy(
        splits["train"], feature_cols, action_vocab, args.target, weight_col
    )
    X_val, y_val, _val_weights, val_rows = extract_xy(
        splits["validation"], feature_cols, action_vocab, args.target, weight_col
    )
    X_test, y_test, _test_weights, test_rows = extract_xy(
        splits["test"], feature_cols, action_vocab, args.target, weight_col
    )
    if not X_train:
        raise RuntimeError("Training split has no valid target rows")

    LOGGER.info(
        "Training %s target=%s rows train=%d validation=%d test=%d features=%d actions=%d",
        args.model, args.target, len(X_train), len(X_val), len(X_test),
        len(feature_cols), len(action_vocab),
    )
    model = get_model(args.model, args.seed)
    fit_started = time.perf_counter()
    fit_kwargs: Dict[str, Any] = {}
    if weight_col:
        fit_kwargs["sample_weight"] = train_weights
    sample_weight_used = bool(fit_kwargs)
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        if fit_kwargs:
            LOGGER.warning(
                "%s does not accept sample_weight; fitting without weights",
                type(model).__name__,
            )
            model.fit(X_train, y_train)
            sample_weight_used = False
        else:
            raise
    fit_seconds = time.perf_counter() - fit_started
    LOGGER.info("Model fit completed in %.3f seconds", fit_seconds)

    metrics: Dict[str, Any] = {
        "target": args.target,
        "model_type_requested": args.model,
        "model_class": type(model).__name__,
        "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "feature_count": len(feature_cols),
        "action_count": len(action_vocab),
        "sample_weight_col": weight_col,
        "sample_weighted_training": sample_weight_used,
        "train_weight_sum": sum(train_weights) if weight_col else None,
        "train_zero_weight_rows": (
            sum(1 for weight in train_weights if weight <= 0.0)
            if weight_col else None
        ),
        "train": regression_metrics(model, X_train, y_train),
        "validation": regression_metrics(model, X_val, y_val),
        "test": regression_metrics(model, X_test, y_test),
        "validation_ranking": ranking_metrics(
            model, val_rows, feature_cols, action_vocab, args.target
        ),
        "test_ranking": ranking_metrics(
            model, test_rows, feature_cols, action_vocab, args.target
        ),
    }
    if hasattr(model, "n_iter_"):
        metrics["iterations_completed"] = int(model.n_iter_)

    try:
        import joblib
        joblib.dump(model, model_dir / "sl_reward_model.joblib")
    except Exception:
        with (model_dir / "sl_reward_model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

    (model_dir / "sl_action_vocab.json").write_text(
        json.dumps(action_vocab, indent=2, sort_keys=True) + "\n"
    )
    (model_dir / "sl_pass_list.json").write_text(
        json.dumps(list(action_vocab), indent=2) + "\n"
    )
    (model_dir / "sl_feature_columns.json").write_text(
        json.dumps(
            {
                "feature_cols": feature_cols,
                "uses_normalized_features": args.use_normalized,
                "action_encoding": "one_hot",
                "target": args.target,
                "sample_weight_col": weight_col,
                "input_dimension": len(feature_cols) + len(action_vocab),
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    (model_dir / "sl_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    LOGGER.info("Saved model and metrics to %s", model_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True))
    return model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train runtime-aware supervised pass scorer")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--model", default="histgb",
        choices=["lightgbm", "xgboost", "catboost", "randomforest", "rf", "histgb", "mlp"],
    )
    parser.add_argument("--target", default="runtime_improvement_pct", choices=TARGETS)
    parser.add_argument(
        "--use-normalized", action="store_true",
        help="Use norm_pre_* features. Raw features are safer for tree inference and are the default.",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--sample-weight-col",
        default=None,
        help="Optional per-row sample weight column, e.g. reward_reliability_weight.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
