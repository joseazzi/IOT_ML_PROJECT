"""FastAPI inference service for the IoT IDS — Milestone 4.

Endpoints
---------
GET  /health             basic liveness + loaded model info
POST /predict            single feature vector -> classification
POST /predict/batch      list of feature vectors -> list of classifications
GET  /monitoring         full drift/alert snapshot as JSON
GET  /dashboard          small HTML page that polls /monitoring

Run with:
    cd milestone4
    uvicorn app:app --reload
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import config
from monitoring import DriftMonitor, compute_reference_stats
from schemas import (
    BatchFeatureVectors,
    BatchPrediction,
    FeatureVector,
    HealthResponse,
    Prediction,
)

app = FastAPI(
    title="IoT IDS Inference Service",
    description=(
        "Wraps the Milestone 3 model (Random Forest) in a JSON API and "
        "monitors feature drift, predicted class-mix drift, and alert-rate "
        "spikes over a rolling window of recent predictions."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Startup: load model + preprocessing artifacts, compute reference stats
# ---------------------------------------------------------------------
class ModelBundle:
    """Holds everything loaded at startup so request handlers are cheap."""

    def __init__(self) -> None:
        self.model = joblib.load(config.MODEL_PATH, mmap_mode="r")
        self.imputer = joblib.load(config.IMPUTER_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
        with open(config.FEATURE_NAMES_PATH) as f:
            self.feature_names: List[str] = json.load(f)

        self.class_names: List[str] = list(self.label_encoder.classes_)
        self.n_features = len(self.feature_names)
        self.n_classes = len(self.class_names)

        # Reference distributions for drift (computed from training arrays)
        X_train = np.load(config.X_TRAIN_PATH, mmap_mode="r")
        y_train = np.load(config.Y_TRAIN_PATH, mmap_mode="r")
        means, stds, shares = compute_reference_stats(
            X_train, y_train, self.n_classes
        )
        self.drift_monitor = DriftMonitor(
            train_feature_means=means,
            train_feature_stds=stds,
            train_class_shares=shares,
            feature_names=self.feature_names,
            class_names=self.class_names,
        )


bundle: ModelBundle | None = None


@app.on_event("startup")
def _startup() -> None:
    global bundle
    bundle = ModelBundle()
    print(f"Loaded {type(bundle.model).__name__} from {config.MODEL_PATH}")
    print(f"  features: {bundle.n_features}")
    print(f"  classes:  {bundle.class_names}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _records_to_matrix(records: List[Dict[str, float]]) -> np.ndarray:
    """Turn a list of {feature_name: value} dicts into a 2D numpy array in
    the *training* column order. Missing keys become NaN (imputed downstream)."""
    assert bundle is not None
    matrix = np.full((len(records), bundle.n_features), np.nan, dtype=float)
    for i, rec in enumerate(records):
        for j, name in enumerate(bundle.feature_names):
            if name in rec:
                try:
                    matrix[i, j] = float(rec[name])
                except (TypeError, ValueError):
                    # Leave as NaN -> imputer will handle
                    pass
    return matrix


def _preprocess(raw: np.ndarray) -> np.ndarray:
    """Apply the same imputer + scaler fit at training time."""
    assert bundle is not None
    raw = np.where(np.isinf(raw), np.nan, raw)  # treat inf as NaN
    imputed = bundle.imputer.transform(raw)
    scaled = bundle.scaler.transform(imputed)
    return scaled


def _log_predictions(records: List[Dict[str, float]], preds: List[Prediction]) -> None:
    """Append prediction events to predictions.jsonl (one JSON per line)."""
    now = datetime.utcnow().isoformat() + "Z"
    with open(config.PREDICTION_LOG, "a") as f:
        for rec, pred in zip(records, preds):
            entry = {
                "timestamp": now,
                "predicted_class": pred.predicted_class,
                "predicted_index": pred.predicted_index,
                "is_attack": pred.is_attack,
                "top_proba": max(pred.probabilities.values()),
            }
            f.write(json.dumps(entry) + "\n")


def _predict_matrix(raw: np.ndarray) -> List[Prediction]:
    """Shared prediction path for single and batch endpoints."""
    assert bundle is not None
    scaled = _preprocess(raw)

    if hasattr(bundle.model, "predict_proba"):
        probas = bundle.model.predict_proba(scaled)
    else:
        # Fallback: one-hot from hard predictions
        hard = bundle.model.predict(scaled)
        probas = np.eye(bundle.n_classes)[hard]

    preds_idx = probas.argmax(axis=1)

    # Record into the drift monitor
    bundle.drift_monitor.record(scaled, preds_idx)

    out: List[Prediction] = []
    for i, idx in enumerate(preds_idx):
        name = bundle.class_names[int(idx)]
        out.append(
            Prediction(
                predicted_class=name,
                predicted_index=int(idx),
                is_attack=(int(idx) != config.BENIGN_CLASS),
                probabilities={
                    bundle.class_names[k]: float(probas[i, k])
                    for k in range(bundle.n_classes)
                },
            )
        )
    return out


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return HealthResponse(
        status="ok",
        model_loaded=True,
        n_features=bundle.n_features,
        n_classes=bundle.n_classes,
        classes=bundle.class_names,
    )


@app.post("/predict", response_model=Prediction)
def predict(payload: FeatureVector) -> Prediction:
    """Classify one network flow record."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw = _records_to_matrix([payload.features])
    preds = _predict_matrix(raw)
    _log_predictions([payload.features], preds)
    return preds[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(payload: BatchFeatureVectors) -> BatchPrediction:
    """Classify a micro-batch of flow records (for streaming simulation)."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw = _records_to_matrix(payload.records)
    preds = _predict_matrix(raw)
    _log_predictions(payload.records, preds)

    alert_count = sum(1 for p in preds if p.is_attack)
    return BatchPrediction(
        predictions=preds,
        alert_count=alert_count,
        alert_rate=alert_count / len(preds),
    )


@app.get("/monitoring")
def monitoring() -> Dict:
    """Current drift + alert-rate snapshot, and write it to the drift log."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    snap = bundle.drift_monitor.snapshot()
    bundle.drift_monitor.log_snapshot(snap)
    return snap


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Tiny HTML page that polls /monitoring and renders a few charts."""
    with open(config.HERE / "templates" / "dashboard.html") as f:
        return HTMLResponse(f.read())
