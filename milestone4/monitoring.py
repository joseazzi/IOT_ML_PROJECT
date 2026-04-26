"""Drift and alert-rate monitoring.

Implements the three signals from the brief's Step 7:
  1. Feature drift (distribution of incoming features vs training).
  2. Predicted-class-mix drift (how the class histogram shifts over time).
  3. Alert-rate spikes (sudden jump in non-benign predictions).

Everything is kept in memory in a fixed-size sliding window. No database.
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Tuple

import numpy as np

from config import (
    ALERT_SPIKE_RATIO,
    BENIGN_CLASS,
    CLASS_DRIFT_THRESHOLD,
    DRIFT_LOG,
    FEATURE_DRIFT_SIGMA,
    WINDOW_SIZE,
)


class DriftMonitor:
    """Rolling-window drift monitor.

    Parameters
    ----------
    train_feature_means, train_feature_stds
        Per-feature mean and std computed over the (scaled) training set.
        Used as the reference distribution for feature drift.
    train_class_shares
        Class-share vector from the training set, indexed by class id.
        Used as the reference for class-mix drift.
    feature_names
        Names of the features in the same order as the mean/std arrays.
    class_names
        Class names in class-id order.
    """

    def __init__(
        self,
        train_feature_means: np.ndarray,
        train_feature_stds: np.ndarray,
        train_class_shares: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
    ) -> None:
        self.train_feature_means = train_feature_means
        # Guard against zero std (shouldn't happen after StandardScaler but just in case)
        self.train_feature_stds = np.where(
            train_feature_stds > 1e-9, train_feature_stds, 1.0
        )
        self.train_class_shares = train_class_shares
        self.feature_names = feature_names
        self.class_names = class_names

        # Baseline alert rate = share of non-benign in training
        self.baseline_alert_rate = float(1.0 - train_class_shares[BENIGN_CLASS])

        # Rolling windows
        self._recent_features: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self._recent_preds: Deque[int] = deque(maxlen=WINDOW_SIZE)

        # Cumulative counters (never reset)
        self.total_predictions = 0
        self.total_alerts = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def record(self, X_scaled: np.ndarray, y_pred: np.ndarray) -> None:
        """Append one or more (scaled_features, predicted_class) pairs."""
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)
            y_pred = np.atleast_1d(y_pred)

        for row, cls in zip(X_scaled, y_pred):
            self._recent_features.append(row)
            self._recent_preds.append(int(cls))
            self.total_predictions += 1
            if int(cls) != BENIGN_CLASS:
                self.total_alerts += 1

    # ------------------------------------------------------------------
    # Drift computations
    # ------------------------------------------------------------------
    def feature_drift(self) -> Dict:
        """Per-feature standardised mean shift over the rolling window."""
        n = len(self._recent_features)
        if n < 10:
            return {
                "window_size": n,
                "sufficient_data": False,
                "per_feature_sigma": {},
                "n_drifting_features": 0,
                "max_drifting_feature": None,
            }

        window = np.asarray(self._recent_features)
        window_mean = window.mean(axis=0)

        # In the SCALED space, training mean ~= 0 and training std ~= 1.
        # We still compute the shift explicitly so this works if the caller
        # passed raw (non-scaled) features.
        shift_sigma = np.abs(window_mean - self.train_feature_means) / self.train_feature_stds

        drifting_mask = shift_sigma > FEATURE_DRIFT_SIGMA
        per_feature = {
            name: float(sigma)
            for name, sigma in zip(self.feature_names, shift_sigma)
        }

        max_idx = int(np.argmax(shift_sigma))
        max_feature = {
            "name": self.feature_names[max_idx],
            "shift_sigma": float(shift_sigma[max_idx]),
        }

        return {
            "window_size": n,
            "sufficient_data": True,
            "threshold_sigma": FEATURE_DRIFT_SIGMA,
            "per_feature_sigma": per_feature,
            "n_drifting_features": int(drifting_mask.sum()),
            "max_drifting_feature": max_feature,
        }

    def class_mix_drift(self) -> Dict:
        """Compare predicted class shares in the window vs training shares."""
        n = len(self._recent_preds)
        if n < 10:
            return {
                "window_size": n,
                "sufficient_data": False,
                "per_class": {},
                "max_abs_diff": 0.0,
                "drift_detected": False,
            }

        counts = np.bincount(
            np.asarray(self._recent_preds, dtype=int),
            minlength=len(self.class_names),
        )
        shares = counts / counts.sum()

        per_class = {}
        for i, name in enumerate(self.class_names):
            per_class[name] = {
                "recent_share": float(shares[i]),
                "training_share": float(self.train_class_shares[i]),
                "abs_diff": float(abs(shares[i] - self.train_class_shares[i])),
            }

        max_diff = max(c["abs_diff"] for c in per_class.values())

        return {
            "window_size": n,
            "sufficient_data": True,
            "threshold": CLASS_DRIFT_THRESHOLD,
            "per_class": per_class,
            "max_abs_diff": float(max_diff),
            "drift_detected": bool(max_diff > CLASS_DRIFT_THRESHOLD),
        }

    def alert_rate(self) -> Dict:
        """Recent alert rate vs the training-time baseline."""
        n = len(self._recent_preds)
        if n < 10:
            return {
                "window_size": n,
                "sufficient_data": False,
                "recent_alert_rate": 0.0,
                "baseline_alert_rate": self.baseline_alert_rate,
                "ratio": 0.0,
                "spike_detected": False,
            }

        preds = np.asarray(self._recent_preds)
        recent = float((preds != BENIGN_CLASS).mean())
        # Guard against div-by-zero if baseline is zero
        baseline = max(self.baseline_alert_rate, 1e-6)
        ratio = recent / baseline

        return {
            "window_size": n,
            "sufficient_data": True,
            "spike_ratio_threshold": ALERT_SPIKE_RATIO,
            "recent_alert_rate": recent,
            "baseline_alert_rate": self.baseline_alert_rate,
            "ratio": float(ratio),
            "spike_detected": bool(ratio > ALERT_SPIKE_RATIO),
        }

    # ------------------------------------------------------------------
    # Combined snapshot + persistence
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict:
        """Full monitoring snapshot — returned by GET /monitoring."""
        fd = self.feature_drift()
        cd = self.class_mix_drift()
        ar = self.alert_rate()

        any_drift = (
            (fd.get("n_drifting_features", 0) > 0)
            or cd.get("drift_detected", False)
            or ar.get("spike_detected", False)
        )

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_predictions": self.total_predictions,
            "total_alerts": self.total_alerts,
            "window_size_config": WINDOW_SIZE,
            "status": "DRIFT" if any_drift else "OK",
            "feature_drift": fd,
            "class_mix_drift": cd,
            "alert_rate": ar,
        }

    def log_snapshot(self, snap: Dict) -> None:
        """Append a compact snapshot to drift.jsonl."""
        compact = {
            "timestamp": snap["timestamp"],
            "status": snap["status"],
            "total_predictions": snap["total_predictions"],
            "total_alerts": snap["total_alerts"],
            "n_drifting_features": snap["feature_drift"].get("n_drifting_features", 0),
            "class_max_abs_diff": snap["class_mix_drift"].get("max_abs_diff", 0.0),
            "alert_rate_ratio": snap["alert_rate"].get("ratio", 0.0),
        }
        with open(DRIFT_LOG, "a") as f:
            f.write(json.dumps(compact) + "\n")


# ---------------------------------------------------------------------
# Reference-stats computation (called once at app startup)
# ---------------------------------------------------------------------
def compute_reference_stats(
    X_train_scaled: np.ndarray, y_train: np.ndarray, n_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (feature_means, feature_stds, class_shares) for the training set."""
    feat_means = X_train_scaled.mean(axis=0)
    feat_stds = X_train_scaled.std(axis=0)
    counts = np.bincount(y_train.astype(int), minlength=n_classes)
    class_shares = counts / counts.sum()
    return feat_means, feat_stds, class_shares
