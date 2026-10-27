from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from libpysal.weights import DistanceBand
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

from ._spatial_features import build_distance_weights as build_radii_weights
from ._spatial_features import compute_lag_features


LOGGER_NAME = "land_value_training"


@dataclass
class TrainingConfig:
    feature_cols: Sequence[str]
    cat_features: Sequence[str]
    radius_list: Sequence[int]
    target_col: str = "log_total_price"
    group_col: str = "district_name"
    loss_function: str = "MAE"
    eval_metric: str = "MAE"

    iterations: int = 1500
    od_wait: int = 300
    seed: int = 42

    params: Optional[Mapping[str, Any]] = None

    # как часто CatBoost печатает прогресс (итерации). 0 = тихо
    catboost_verbose: int = 200  # например 100/200


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        from tqdm.auto import tqdm as _tqdm

        try:
            _tqdm.write(self.format(record))
        except Exception:  # pragma: no cover
            self.handleError(record)


def setup_logger(log_path: Path | str = "training.log", *, console_level: int = logging.INFO) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if not any(isinstance(h, TqdmLoggingHandler) for h in logger.handlers):
        sh = TqdmLoggingHandler()
        sh.setLevel(console_level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def to_original_scale(y: np.ndarray, *, target_col: str) -> np.ndarray:
    # Training uses log1p-based targets (log_*), metrics should be reported in rubles.
    if target_col.startswith("log_"):
        return np.expm1(y)
    return y


def wape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    denom = float(np.sum(np.abs(y_true[mask])))
    if denom <= eps:
        return float("nan")
    num = float(np.sum(np.abs(y_true[mask] - y_pred[mask])))
    return num / denom


def mpe(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not np.any(mask):
        return float("nan")
    # Positive means overprediction, consistent with `mean_error`.
    return float(np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    denom = np.abs(y_t) + np.abs(y_p)
    valid = denom > eps
    if not np.any(valid):
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_t[valid] - y_p[valid]) / denom[valid]))


def mdape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not np.any(mask):
        return float("nan")
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.median(ape))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.clip(y_true, a_min=0.0, a_max=None)
    y_p = np.clip(y_pred, a_min=0.0, a_max=None)
    return float(np.sqrt(np.mean((np.log1p(y_t) - np.log1p(y_p)) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, target_col: str) -> Dict[str, float]:
    y_true_eval = to_original_scale(y_true, target_col=target_col)
    y_pred_eval = to_original_scale(y_pred, target_col=target_col)

    metrics: Dict[str, float] = {
        "r2": float(r2_score(y_true_eval, y_pred_eval)),
        "mae": float(mean_absolute_error(y_true_eval, y_pred_eval)),
        "rmse": float(rmse(y_true_eval, y_pred_eval)),
        "mape_ratio": float(mape(y_true_eval, y_pred_eval)),
        "wape_ratio": float(wape(y_true_eval, y_pred_eval)),
    }
    return metrics


def format_metrics_report(train_metrics: Mapping[str, float], test_metrics: Mapping[str, float]) -> str:
    return (
        "Train metrics:\n"
        f"MAPE: {train_metrics['mape_ratio']:.6f} ({train_metrics['mape_ratio'] * 100.0:.2f}%)\n"
        f"WAPE: {train_metrics['wape_ratio']:.6f} ({train_metrics['wape_ratio'] * 100.0:.2f}%)\n"
        f"MAE:  {train_metrics['mae']:.6f}\n"
        f"RMSE: {train_metrics['rmse']:.6f}\n"
        f"R2:   {train_metrics['r2']:.6f}\n\n"
        "Test metrics:\n"
        f"MAPE: {test_metrics['mape_ratio']:.6f} ({test_metrics['mape_ratio'] * 100.0:.2f}%)\n"
        f"WAPE: {test_metrics['wape_ratio']:.6f} ({test_metrics['wape_ratio'] * 100.0:.2f}%)\n"
        f"MAE:  {test_metrics['mae']:.6f}\n"
        f"RMSE: {test_metrics['rmse']:.6f}\n"
        f"R2:   {test_metrics['r2']:.6f}"
    )


def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Bias in target units: >0 means overprediction, <0 means underprediction.
    return float(np.mean(y_pred - y_true))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    denom = float(np.max(y_t) - np.min(y_t))
    if denom <= eps:
        return float("nan")
    return float(rmse(y_t, y_p) / denom)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train: np.ndarray,
    seasonality: int = 1,
    eps: float = 1e-12,
) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_tr = y_train[np.isfinite(y_train)]
    if len(y_tr) <= seasonality:
        return float("nan")
    scale = float(np.mean(np.abs(y_tr[seasonality:] - y_tr[:-seasonality])))
    if scale <= eps:
        return float("nan")
    return float(np.mean(np.abs(y_t - y_p)) / scale)


def residuals_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_col: str,
) -> pd.DataFrame:
    y_true_eval = to_original_scale(y_true, target_col=target_col)
    y_pred_eval = to_original_scale(y_pred, target_col=target_col)
    residual = y_true_eval - y_pred_eval
    return pd.DataFrame({"y_true": y_true_eval, "y_pred": y_pred_eval, "residual": residual})


def plot_residual_diagnostics(
    residuals_df: pd.DataFrame,
    *,
    bins: int = 50,
    figsize: Tuple[int, int] = (14, 5),
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(residuals_df["residual"].to_numpy(), bins=bins, alpha=0.9, edgecolor="white")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals Distribution")
    axes[0].set_xlabel("Residual (y_true - y_pred)")
    axes[0].set_ylabel("Count")

    axes[1].scatter(
        residuals_df["y_pred"].to_numpy(),
        residuals_df["residual"].to_numpy(),
        s=12,
        alpha=0.35,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual (y_true - y_pred)")

    fig.tight_layout()
    return fig, axes


def groups_from_column(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}' not found in DataFrame.")
    return df[group_col].astype("string").fillna("missing_group")


def prep_cat_inplace(df: pd.DataFrame, cat_cols: Sequence[str]) -> None:
    for c in cat_cols:
        if c not in df.columns:
            continue
        df[c] = df[c].astype("string").fillna("missing")


def add_lags_full_df(
    df: pd.DataFrame,
    *,
    weights: Mapping[int, DistanceBand],
    numeric_cols: Sequence[str],
) -> pd.DataFrame:
    features = [feature for feature in numeric_cols if feature in df.columns]
    lags = compute_lag_features(df, features, weights)
    return pd.concat([df.copy(), lags], axis=1)


def feature_names(df: pd.DataFrame, base_cols: Sequence[str], target_col: str) -> List[str]:
    extra = [c for c in df.columns if c.startswith("lag") or c.startswith("n_neighbors_")]
    final = list(dict.fromkeys(list(base_cols) + extra))
    return [c for c in final if c != target_col and c in df.columns]


def build_lagged_fold_frames(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    numeric_feats: Sequence[str],
    radius_list: Sequence[int],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Leakage-safe setup: train/test lags are computed independently per fold.
    tr_df = df_train_base.copy()
    te_df = df_test_base.copy()

    prep_cat_inplace(tr_df, cat_features)
    prep_cat_inplace(te_df, cat_features)

    tr_weights = build_radii_weights(tr_df, radius_list)
    te_weights = build_radii_weights(te_df, radius_list)

    tr_lag = add_lags_full_df(tr_df, weights=tr_weights, numeric_cols=numeric_feats)
    te_lag = add_lags_full_df(te_df, weights=te_weights, numeric_cols=numeric_feats)

    base_cols_for_model = list(dict.fromkeys(list(feature_cols) + list(cat_features)))
    tr_feats = feature_names(tr_lag, base_cols_for_model, target_col)
    te_feats = feature_names(te_lag, base_cols_for_model, target_col)
    feats = [c for c in tr_feats if c in te_feats]
    if not feats:
        raise RuntimeError("No common feature columns between train and test fold after lag generation.")

    return tr_lag, te_lag, feats


def fit_and_eval(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    *,
    target_col: str,
    loss_function: str,
    eval_metric: str,
    cat_features: Sequence[str],
    feats: Sequence[str],
    params: Mapping[str, Any],
    iterations: int,
    od_wait: int,
    seed: int,
    logger: logging.Logger,
) -> Tuple[CatBoostRegressor, Dict[str, float], int]:
    cat_cols_used = [c for c in cat_features if c in feats]

    train_pool = Pool(
        df_train_base[feats], label=df_train_base[target_col], cat_features=cat_cols_used, feature_names=feats
    )
    test_pool = Pool(
        df_test_base[feats], label=df_test_base[target_col], cat_features=cat_cols_used, feature_names=feats
    )

    model = CatBoostRegressor(
        loss_function=loss_function,
        eval_metric=eval_metric,
        iterations=iterations,
        od_type="Iter",
        od_wait=od_wait,
        bootstrap_type="Bayesian",
        grow_policy="SymmetricTree",
        random_seed=seed,
        verbose=0,  # outer folds тоже лучше без спама
        **params,
    )
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=od_wait, use_best_model=True)

    y_train = df_train_base[target_col].to_numpy()
    y_pred_train = model.predict(df_train_base[feats])
    y_true = df_test_base[target_col].to_numpy()
    y_pred = model.predict(df_test_base[feats])
    y_train_eval = to_original_scale(y_train, target_col=target_col)

    train_metrics = regression_metrics(y_train, y_pred_train, target_col=target_col)
    test_metrics = regression_metrics(y_true, y_pred, target_col=target_col)

    y_true_eval = to_original_scale(y_true, target_col=target_col)
    y_pred_eval = to_original_scale(y_pred, target_col=target_col)
    mpe_ratio = float(mpe(y_true_eval, y_pred_eval))
    smape_ratio = float(smape(y_true_eval, y_pred_eval))
    mdape_ratio = float(mdape(y_true_eval, y_pred_eval))

    m: Dict[str, float] = {
        "r2": test_metrics["r2"],
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "nrmse": float(nrmse(y_true_eval, y_pred_eval)),
        "mean_error": float(mean_error(y_true_eval, y_pred_eval)),
        # Percentage metrics are reported in percent.
        "mpe": float(mpe_ratio * 100.0),
        "mape": float(test_metrics["mape_ratio"] * 100.0),
        "wape": float(test_metrics["wape_ratio"] * 100.0),
        "smape": float(smape_ratio * 100.0),
        "mdape": float(mdape_ratio * 100.0),
        "mase": float(mase(y_true_eval, y_pred_eval, y_train=y_train_eval)),
    }
    if target_col.startswith("log_"):
        m["rmsle"] = float(rmsle(y_true_eval, y_pred_eval))

    best_it = model.get_best_iteration()
    if best_it is None or best_it <= 0:
        best_it = model.tree_count_

    logger.info("%s\nbest_iter=%d", format_metrics_report(train_metrics, test_metrics), int(best_it))
    return model, m, int(best_it)


def run_training(
    blocks: pd.DataFrame,
    config: TrainingConfig,
    *,
    log_path: Path | str = "training.log",
    console_level: int = logging.INFO,
) -> Tuple[CatBoostRegressor, Dict[str, Any]]:
    logger = setup_logger(log_path, console_level=console_level)
    logger.info("Starting district leave-one-group-out training")

    target_col = config.target_col
    group_col = config.group_col
    cat_features = list(config.cat_features)
    numeric_feats = [
        c for c in config.feature_cols if c not in cat_features and c != target_col and c in blocks.columns
    ]
    params = dict(config.params or {})

    groups_all = groups_from_column(blocks, group_col)
    unique_groups = groups_all.drop_duplicates().sort_values().tolist()
    if len(unique_groups) < 2:
        raise ValueError(
            f"Need at least 2 unique districts in '{group_col}' for leave-one-group-out training; "
            f"got {len(unique_groups)}."
        )
    logger.info("Using leave-one-group-out by '%s': %d districts", group_col, len(unique_groups))
    print(f"[run_training] leave-one-group-out by '{group_col}', folds={len(unique_groups)}")

    fold_metrics: List[Dict[str, float]] = []
    best_iters: List[int] = []
    fold_details: List[Dict[str, Any]] = []

    for fold_i, test_group in enumerate(tqdm(unique_groups, desc="DistrictCV", total=len(unique_groups)), 1):
        test_mask = groups_all == test_group
        df_train_base = blocks.loc[~test_mask].copy()
        df_test_base = blocks.loc[test_mask].copy()

        if df_train_base.empty or df_test_base.empty:
            raise RuntimeError(f"Empty train/test split for district '{test_group}' in fold {fold_i}.")

        df_train, df_test, feats = build_lagged_fold_frames(
            df_train_base,
            df_test_base,
            feature_cols=config.feature_cols,
            cat_features=cat_features,
            numeric_feats=numeric_feats,
            radius_list=config.radius_list,
            target_col=target_col,
        )
        logger.info(
            "Fold %d/%d: test_district='%s', train=%d, test=%d, feats=%d",
            fold_i,
            len(unique_groups),
            test_group,
            len(df_train),
            len(df_test),
            len(feats),
        )

        model, m, best_it = fit_and_eval(
            df_train,
            df_test,
            target_col=target_col,
            loss_function=config.loss_function,
            eval_metric=config.eval_metric,
            cat_features=cat_features,
            feats=feats,
            params=params,
            iterations=config.iterations,
            od_wait=config.od_wait,
            seed=config.seed,
            logger=logger,
        )

        fold_metrics.append(m)
        best_iters.append(best_it)
        fold_details.append(
            {
                "fold": fold_i,
                "test_district": str(test_group),
                "n_train": int(len(df_train)),
                "n_test": int(len(df_test)),
                "best_iter": int(best_it),
            }
        )

    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean(numeric_only=True).to_dict()
    std_metrics = metrics_df.std(numeric_only=True).to_dict()

    summary: Dict[str, Any] = {}
    for k, v in mean_metrics.items():
        summary[k] = float(v)
        summary[f"{k}_std"] = float(std_metrics.get(k, np.nan))
    summary["folds"] = fold_details
    summary["n_folds"] = len(unique_groups)

    logger.info("DistrictCV mean metrics=%s", summary)

    final_iterations = int(np.median(best_iters)) if best_iters else config.iterations
    final_iterations = max(200, final_iterations)
    logger.info("Training final model on FULL dataset: iterations=%d params=%s", final_iterations, params)

    base_df = blocks.copy()
    prep_cat_inplace(base_df, cat_features)
    weights = build_radii_weights(base_df, config.radius_list)
    blocks_lag = add_lags_full_df(base_df, weights=weights, numeric_cols=numeric_feats)
    base_cols_for_model = list(dict.fromkeys(list(config.feature_cols) + cat_features))
    feats = feature_names(blocks_lag, base_cols_for_model, target_col)
    cat_cols_used = [c for c in cat_features if c in feats]

    full_pool = Pool(blocks_lag[feats], label=blocks_lag[target_col], cat_features=cat_cols_used, feature_names=feats)

    final_model = CatBoostRegressor(
        loss_function=config.loss_function,
        eval_metric=config.eval_metric,
        iterations=final_iterations,
        bootstrap_type="Bayesian",
        grow_policy="SymmetricTree",
        random_seed=config.seed,
        verbose=config.catboost_verbose,  # вот тут будет прогресс по итерациям
        **params,
    )
    final_model.fit(full_pool)

    final_pred_raw = np.asarray(final_model.predict(blocks_lag[feats])).reshape(-1)
    final_train_metrics = regression_metrics(
        blocks_lag[target_col].to_numpy(),
        final_pred_raw,
        target_col=target_col,
    )
    outer_test_metrics = {
        "mape_ratio": float(summary["mape"] / 100.0),
        "wape_ratio": float(summary["wape"] / 100.0),
        "mae": float(summary["mae"]),
        "rmse": float(summary["rmse"]),
        "r2": float(summary["r2"]),
    }
    report = format_metrics_report(final_train_metrics, outer_test_metrics)
    summary["report"] = report
    logger.warning("%s", report)

    return final_model, summary


__all__ = [
    "TrainingConfig",
    "setup_logger",
    "run_training",
    "build_radii_weights",
    "add_lags_full_df",
    "feature_names",
    "mean_error",
    "residuals_dataframe",
    "plot_residual_diagnostics",
]
