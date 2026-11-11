"""Accuracy metrics for the unmixing against the ground truth."""

from __future__ import annotations

import numpy as np


def fraction_mae(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Mean Absolute Error averaged over the fraction maps (ignores NaN)."""
    est = np.asarray(estimated, dtype=float)
    gt = np.asarray(truth, dtype=float)
    mask = ~np.isnan(est).any(axis=-1)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(est[mask] - gt[mask])))


def per_species_mae(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """MAE for each species (array of length K)."""
    est = np.asarray(estimated, dtype=float)
    gt = np.asarray(truth, dtype=float)
    mask = ~np.isnan(est).any(axis=-1)
    return np.mean(np.abs(est[mask] - gt[mask]), axis=0)
