# IoT Intrusion Detection Project

## Group Members
- Jose Azzi
- Maria Bouchi
- Lynn Mechreck

## Data Source
This project uses a subset of the CICIoT2023 dataset stored in `data/raw/` and consolidated into `data/processed/clean_dataset.csv`. The working dataset contains 1,085,047 labeled network-flow records, 39 input features, and 8 classes: `BenignTraffic`, `DDoS-ICMP_Flood`, `DictionaryBruteForce`, `DoS-SYN_Flood`, `MITM-ArpSpoofing`, `Mirai-greeth_flood`, `Recon-PortScan`, and `SqlInjection`.

## Approach
The goal was to build an IoT intrusion detection system that can both classify network traffic and monitor its behavior after deployment. The project was organized as a pipeline: inspect and clean the dataset, prepare leakage-aware train/validation/test splits, train and compare multiple classifiers, select the best model using security-relevant metrics, then deploy the chosen model behind an API with live monitoring and drift detection.

## Summary of Methods Used
Data preparation started with duplicate removal and replacement of invalid infinite values with missing values. The cleaned data was split using an ordered per-class 70/15/15 train/validation/test protocol. Because explicit timestamps are not available in the cleaned feature files, each class is split by original row order rather than random row shuffling; this keeps adjacent, highly similar records together more often and reduces train/test leakage risk. To keep preprocessing consistent between training and inference, a mean `SimpleImputer`, `StandardScaler`, label encoder, and ordered feature list were fit on the training split only and saved for reuse.

Four models were trained and compared on the validation set: Logistic Regression, Random Forest, HistGradientBoosting, and a PyTorch MLP. Because the dataset is class-imbalanced, class weighting was applied during training and model selection was based on macro F1 rather than accuracy alone. A benign false positive rate was also tracked to ensure the IDS would not generate excessive false alarms in practice.

For deployment, the selected model was wrapped in a FastAPI service in `milestone4/`. The API supports single-record and batch prediction, returns class probabilities, and uses the saved preprocessing artifacts to guarantee the same feature handling at inference time. A monitoring module keeps a rolling window of recent predictions and tracks three operational signals: feature drift, predicted class-mix drift, and alert-rate spikes. A small dashboard and a CSV streaming simulator were added to demonstrate live behavior.

## Summary Result
Random Forest was selected as the best overall model. It achieved validation accuracy of 0.9101, validation macro F1 of 0.7347, and benign false positive rate of 0.0257. On the held-out test set, it achieved accuracy of 0.9158, macro F1 of 0.7602, and benign false positive rate of 0.0219. These results show that the system can classify both benign and attack traffic with strong overall accuracy while keeping false alarms relatively low. The final milestone extends the model into a usable monitoring service that can flag distribution shifts and sudden changes in attack activity during simulated streaming.

## Docker Image
The project is dockerized with a root `Dockerfile`. The image contains the API code, notebooks, trained model artifacts, processed arrays, and raw CSV files needed to run the working IDS service.

The trained Random Forest artifact is large, so Docker Desktop should have at least 6 GB of memory available. If the container exits with code `137`, increase Docker Desktop memory in Settings, then run it again.

Public Docker Hub image name:

```bash
jose442004/iot-intrusion-project:latest
```

The currently pushed Docker Hub image is multi-platform and supports both
`linux/arm64` and `linux/amd64`.

If preprocessing or model artifacts are regenerated after changing the split strategy, rebuild the image before pushing so Docker Hub contains the updated `data/processed/` arrays and `models/` artifacts. The Dockerfile and Compose configuration do not need structural changes for the safer split.

If your Docker Hub username is different, replace `jose442004` in the commands below.

### Run from Docker Hub
After the image is pushed publicly, run:

```bash
docker pull jose442004/iot-intrusion-project:latest
docker run --rm --name iot-ids -p 8000:8000 jose442004/iot-intrusion-project:latest
```

Open these URLs:

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

If port 8000 is already in use:

```bash
docker rm -f iot-ids
docker run --rm --name iot-ids -p 8001:8000 jose442004/iot-intrusion-project:latest
```

### Build and Run Locally
From the project root:

```bash
docker build -t jose442004/iot-intrusion-project:latest .
docker run --rm --name iot-ids -p 8000:8000 jose442004/iot-intrusion-project:latest
```

You can also use Docker Compose:

```bash
docker compose up --build
```

### Simulate Streaming Traffic
With the container running, stream a CSV through the live API:

```bash
docker exec -it iot-ids python simulate_stream.py \
  --source ../data/raw/DDoS-ICMP_Flood.csv \
  --n-records 500 \
  --batch-size 20 \
  --interval 0.5
```

The dashboard at `http://localhost:8000/dashboard` will update from the `/monitoring` endpoint.

### Push to Docker Hub
Create the Docker Hub repository `jose442004/iot-intrusion-project` as a public repository, then run:

```bash
docker login
docker build -t jose442004/iot-intrusion-project:latest .
docker push jose442004/iot-intrusion-project:latest
```
