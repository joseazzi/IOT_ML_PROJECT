# Milestone 4 — Inference Endpoint + Drift Monitoring

Small FastAPI service that loads the Random Forest model saved in Milestone 3
and serves predictions on JSON feature vectors, with three drift signals
monitored over a rolling window of recent predictions.

Maps directly to the brief's **Step 6** (deployment) and **Step 7** (monitoring).

## What this covers

| Brief requirement | Where it lives |
|---|---|
| JSON feature vectors → predictions | `POST /predict` in `app.py` |
| Micro-batch streaming | `POST /predict/batch` in `app.py` |
| Class probabilities returned | Every prediction response |
| Feature-distribution drift | `DriftMonitor.feature_drift()` in `monitoring.py` |
| Predicted class-mix drift | `DriftMonitor.class_mix_drift()` |
| Alert-rate spikes | `DriftMonitor.alert_rate()` |
| Same preprocessing at train/inference time | Loads `imputer.joblib` + `scaler.joblib` saved in Milestone 2 |

## Folder layout

```
milestone4/
├── app.py                 # FastAPI application
├── monitoring.py          # Drift-detection logic (DriftMonitor class)
├── schemas.py             # Pydantic request/response models
├── config.py              # Paths and tunable thresholds
├── templates/
│   └── dashboard.html     # Polls /monitoring and draws charts
├── simulate_stream.py     # Client that streams a CSV through the API
├── requirements.txt
└── logs/                  # Auto-created; per-prediction and per-snapshot logs
    ├── predictions.jsonl
    └── drift.jsonl
```

## Prerequisites

The service expects these artifacts from earlier milestones to exist one
directory level up:

```
../models/best_model.joblib
../models/imputer.joblib
../models/scaler.joblib
../models/label_encoder.joblib
../models/feature_names.json
../data/processed/X_train_scaled.npy
../data/processed/y_train.npy
```

These filenames stay the same after the safer ordered per-class split in
Milestone 2. If the split is regenerated and Milestone 3 is retrained, rebuild
the Docker image so the container includes the updated arrays and model files.

## Install and run

### Run with Docker

The currently pushed Docker Hub image is multi-platform and supports both
`linux/arm64` and `linux/amd64`.

```bash
docker pull jose442004/iot-intrusion-project:latest
docker run --rm --name iot-ids -p 8000:8000 jose442004/iot-intrusion-project:latest
```

Open:

```text
http://localhost:8000/health
http://localhost:8000/docs
http://localhost:8000/dashboard
```

Check from the terminal:

```bash
curl http://localhost:8000/health
```

Stop the container:

```bash
docker stop iot-ids
```

Rerun the container:

```bash
docker run --rm --name iot-ids -p 8000:8000 jose442004/iot-intrusion-project:latest
```

If port 8000 is already in use, remove the old container or use a different
local port:

```bash
docker rm -f iot-ids
docker run --rm --name iot-ids -p 8001:8000 jose442004/iot-intrusion-project:latest
```

If the container exits with code 137, increase Docker Desktop memory to at
least 6 GB and run it again.

### Run locally with Python

```bash
cd milestone4
pip install -r requirements.txt

uvicorn app:app --reload
# -> http://localhost:8000
```

Auto-generated Swagger UI at `http://localhost:8000/docs`.

## Endpoints

### GET /health
```json
{
  "status": "ok",
  "model_loaded": true,
  "n_features": 39,
  "n_classes": 8,
  "classes": ["BenignTraffic", "DDoS-ICMP_Flood", ...]
}
```

### POST /predict
Single flow record.
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": {"Header_Length": 20.0, "Protocol Type": 6, "Rate": 19573.0, ...}}'
```
Response:
```json
{
  "predicted_class": "DoS-SYN_Flood",
  "predicted_index": 3,
  "is_attack": true,
  "probabilities": {"BenignTraffic": 0.01, "DoS-SYN_Flood": 0.98, ...}
}
```
Missing features are treated as NaN and handled by the saved `SimpleImputer`.

### POST /predict/batch
Micro-batch of records — meant for the streaming simulation.
```json
{ "records": [ {"Header_Length": 20.0, ...}, {"Header_Length": 21.4, ...} ] }
```
Response contains a list of `Prediction` objects plus `alert_count` and
`alert_rate` for the batch.

### GET /monitoring
Full drift + alert snapshot. This is also the data source for the dashboard.
```json
{
  "timestamp": "2026-04-17T12:34:56Z",
  "status": "OK",
  "total_predictions": 1240,
  "total_alerts": 187,
  "feature_drift":    { "n_drifting_features": 0,  "max_drifting_feature": {...} },
  "class_mix_drift":  { "max_abs_diff": 0.04,      "drift_detected": false },
  "alert_rate":       { "recent_alert_rate": 0.12, "ratio": 0.8, "spike_detected": false }
}
```

Each call to `/monitoring` also appends a compact snapshot to `logs/drift.jsonl`.

### GET /dashboard
HTML page with live charts of status, class-mix (recent vs training), and
top-10 drifting features. Polls `/monitoring` every 3 seconds.

## Monitoring — how it works

All three signals live in `DriftMonitor` (`monitoring.py`) and share a single
sliding window of `WINDOW_SIZE` (default 500) recent predictions.

- **Feature drift.** For each feature, compute `|mean(window) - mean(train)| / std(train)`.
  Any feature with shift > `FEATURE_DRIFT_SIGMA` (default 2) is flagged as drifting.
  Since features are stored already-scaled (zero mean, unit variance in training), this
  reduces to a simple z-score on the window mean.
- **Class-mix drift.** Compare the class-share histogram in the window to the training
  class-share. Flagged if any class share differs by more than `CLASS_DRIFT_THRESHOLD`
  (default 0.10 = 10 percentage points).
- **Alert-rate spike.** Compute `recent_alert_rate = share of non-benign in window`.
  Flagged if `recent_alert_rate / baseline_alert_rate > ALERT_SPIKE_RATIO` (default 2.0).

Thresholds are all in `config.py` and easy to tune.

## Streaming simulation

`simulate_stream.py` reads a CSV and fires micro-batches at the API. Use it to
see the dashboard react in real time.

### Test streaming with Docker

Start the API container:

```bash
docker run --rm --name iot-ids -p 8000:8000 jose442004/iot-intrusion-project:latest
```

Open the dashboard:

```text
http://localhost:8000/dashboard
```

In a second terminal, run the streaming simulator inside the container:

```bash
docker exec -it iot-ids python simulate_stream.py \
    --source ../data/raw/DDoS-ICMP_Flood.csv \
    --n-records 500 --batch-size 20 --interval 0.5
```

The dashboard should update as batches are sent to `/predict/batch`.

Stop the container when finished:

```bash
docker stop iot-ids
```

### Test streaming locally with Python

```bash
# Mostly benign traffic → dashboard stays in OK state
python simulate_stream.py \
    --source ../data/raw/BenignTraffic.csv \
    --n-records 1000 --batch-size 20 --interval 0.5

# All DDoS traffic → alert rate spikes, class mix drifts
python simulate_stream.py \
    --source ../data/raw/DDoS-ICMP_Flood.csv \
    --n-records 1000 --batch-size 20 --interval 0.5

# A natural mix → watch the predictions settle back to training-like
python simulate_stream.py \
    --source ../data/processed/clean_dataset.csv \
    --n-records 5000 --batch-size 50 --interval 0.2
```



