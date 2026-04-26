"""Pydantic request/response models for the inference API.

Single-record and micro-batch inputs, plus a structured prediction response.
"""
from typing import Dict, List

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """One network-flow record, keyed by feature name.

    Values can be arbitrary numerics; the service re-orders them to the
    training column order before scaling. Missing keys are imputed.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Feature-name -> value mapping for a single flow record",
    )


class BatchFeatureVectors(BaseModel):
    """A micro-batch of flow records."""
    records: List[Dict[str, float]] = Field(
        ...,
        description="List of feature-name -> value mappings",
        min_length=1,
        max_length=10_000,
    )


class Prediction(BaseModel):
    predicted_class: str
    predicted_index: int
    is_attack: bool
    probabilities: Dict[str, float]


class BatchPrediction(BaseModel):
    predictions: List[Prediction]
    alert_count: int
    alert_rate: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    n_classes: int
    classes: List[str]
