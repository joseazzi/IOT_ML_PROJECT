"""Configuration for the Milestone 4 inference service.

Paths are relative to the project root (one level above this folder).
Adjust PROJECT_ROOT if you move things around.
"""
from pathlib import Path

# Folder that *contains* this file -> milestone4/
HERE = Path(__file__).resolve().parent

# Project root -> parent of milestone4/
PROJECT_ROOT = HERE.parent

# Artifacts produced by Milestone 2 (preprocessing) and Milestone 3 (training)
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH         = MODELS_DIR / "best_model.joblib"
IMPUTER_PATH       = MODELS_DIR / "imputer.joblib"
SCALER_PATH        = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# Training arrays — used once at startup to compute reference distributions
X_TRAIN_PATH = DATA_DIR / "X_train_scaled.npy"
Y_TRAIN_PATH = DATA_DIR / "y_train.npy"

# Where the service writes logs
LOGS_DIR = HERE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

PREDICTION_LOG = LOGS_DIR / "predictions.jsonl"
DRIFT_LOG      = LOGS_DIR / "drift.jsonl"

# Drift monitoring parameters
WINDOW_SIZE = 500          # rolling window of recent predictions
FEATURE_DRIFT_SIGMA = 2.0  # |mean_shift| / train_std > this => feature drifting
CLASS_DRIFT_THRESHOLD = 0.10   # max absolute class-share difference vs reference
ALERT_SPIKE_RATIO = 2.0    # recent_alert_rate / baseline > this => spike

# Benign class index (LabelEncoder sorts alphabetically so BenignTraffic is index 0)
BENIGN_CLASS = 0
