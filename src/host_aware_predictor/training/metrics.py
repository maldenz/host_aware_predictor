"""Regression metrics for concat-head expression prediction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _as_1d(array: object) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]

    if true.size == 0:
        return {
            "n": 0.0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    residual = pred - true
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float("nan") if denom == 0.0 else float(1.0 - np.sum(residual**2) / denom)

    true_rank = pd.Series(true).rank(method="average").to_numpy(dtype=np.float64)
    pred_rank = pd.Series(pred).rank(method="average").to_numpy(dtype=np.float64)

    return {
        "n": float(true.size),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "r2": r2,
        "pearson": _safe_corr(true, pred),
        "spearman": _safe_corr(true_rank, pred_rank),
    }
