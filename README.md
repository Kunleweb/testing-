# Fraudlytics — Real-Time Fraud Detection Pipeline

An end-to-end MLOps system that ingests live transaction data from Kafka, trains an XGBoost fraud detection model on a daily schedule via Apache Airflow, serves real-time predictions through a dedicated inference consumer, and visualises results in a live Grafana dashboard — all containerised with Docker Compose.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Getting Started](#getting-started)
- [Service URLs](#service-urls)
- [Screenshots](#screenshots)
- [System Requirements](#system-requirements)
- [Configuration](#configuration)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PRODUCTION LAYER                            │
│                                                                         │
│   Python Producer (×2 replicas)                                         │
│   └── Synthetic transaction generator with realistic fraud patterns     │
│       └── Confluent Cloud Kafka  ──►  transactions topic                │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
          ┌────────────────────────┴──────────────────────────┐
          │                                                   │
          ▼                                                   ▼
┌─────────────────────┐                          ┌───────────────────────┐
│  TRAINING LAYER      │                          │  INFERENCE LAYER      │
│                      │                          │                       │
│  Airflow DAG         │                          │  Fraud Consumer       │
│  (daily @ 03:00 UTC) │                          │  ├── Loads champion   │
│  ├── Validate env    │                          │  │   model from       │
│  ├── Consume Kafka   │                          │  │   MLflow           │
│  ├── Feature eng.    │                          │  ├── Engineers        │
│  ├── SMOTE + XGBoost │                          │  │   features         │
│  ├── Threshold opt.  │                          │  ├── Scores each txn  │
│  ├── Log to MLflow   │                          │  └── Publishes to     │
│  └── Promote champion│                          │      fraud_predictions│
└──────────┬───────────┘                          └──────────┬────────────┘
           │                                                  │
           ▼                                                  ▼
┌─────────────────────────────┐              ┌───────────────────────────┐
│  MODEL REGISTRY LAYER        │              │  OBSERVABILITY LAYER       │
│                              │              │                           │
│  MLflow Server  (:5500)      │              │  Predictions Sink         │
│  └── Experiment tracking     │              │  └── Kafka consumer       │
│  └── Model versioning        │◄─────────── │      writing to           │
│  └── Champion alias          │              │      PostgreSQL            │
│                              │              │                           │
│  MinIO  (:9000)              │              │  Grafana  (:3001)         │
│  └── Model artifacts (S3)    │              │  └── Live dashboard       │
│  └── Plots, metrics          │              │  └── Fraud rate, alerts   │
└─────────────────────────────┘              └───────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data streaming | Confluent Cloud Kafka | Transaction ingestion & prediction output |
| Orchestration | Apache Airflow 3.0 (Celery) | Scheduled model training pipeline |
| ML Framework | XGBoost + scikit-learn | Fraud classification |
| Class balancing | SMOTE (imbalanced-learn) | Handle ~0.3% fraud rate |
| Hyperparameter tuning | RandomizedSearchCV | Automated model optimisation |
| Experiment tracking | MLflow 3.x | Metrics, artifacts, model registry |
| Artifact storage | MinIO (S3-compatible) | Model files, plots, thresholds |
| Metadata database | PostgreSQL 16 | Airflow, MLflow, predictions storage |
| Task queue | Redis + Celery | Airflow worker message bus |
| Inference serving | Python Kafka consumer | Real-time transaction scoring |
| Monitoring | Grafana + PostgreSQL | Live fraud dashboard |
| Containerisation | Docker Compose | Full stack orchestration |

---

## Project Structure

```
fraudlytics/
├── src/
│   ├── docker-compose.yml          # Full infrastructure definition (12+ services)
│   ├── config.yaml                 # MLflow + Kafka configuration
│   ├── .env                        # Secrets (Kafka credentials, MinIO, Fernet key)
│   │
│   ├── airflow/
│   │   ├── Dockerfile              # Airflow image with ML dependencies
│   │   └── requirements.txt        # xgboost, mlflow, sklearn, imblearn, etc.
│   │
│   ├── dags/
│   │   ├── fraud_detection_training_dag.py   # Airflow DAG (3 tasks + promotion gate)
│   │   └── fraud_detection_training.py       # Training class (Kafka → XGBoost → MLflow)
│   │
│   ├── producer/
│   │   ├── main.py                 # Synthetic transaction generator
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── consumer/
│   │   ├── main.py                 # Real-time inference consumer
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── predictions_sink/
│   │   ├── main.py                 # Writes fraud_predictions → PostgreSQL
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── mlflow/
│   │   ├── Dockerfile              # MLflow server image
│   │   └── requirements.txt
│   │
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/        # Auto-provisioned PostgreSQL datasource
│       │   └── dashboards/         # Dashboard loader config
│       └── dashboards/
│           └── fraud_detection.json  # Pre-built fraud monitoring dashboard
```

---

## ML Pipeline

### Training DAG (daily @ 03:00 UTC)

```
validate_environment → execute_training → promote_model → cleanup_resources
```

| Task | Description |
|---|---|
| `validate_environment` | Confirms config and secrets are mounted |
| `execute_training` | Full Kafka → feature engineering → XGBoost → MLflow pipeline |
| `promote_model` | Compares new model F1 against current champion; promotes only if better |
| `cleanup_resources` | Removes temporary artefacts |

### Feature Engineering

Raw Kafka fields are enriched with the following features before training and inference:

| Feature | Source | Description |
|---|---|---|
| `amount` | Raw | Transaction value |
| `log_amount` | Engineered | `log1p(amount)` — reduces skew |
| `hour` | Timestamp | Hour of day (0–23) |
| `day_of_week` | Timestamp | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | Timestamp | Binary flag |
| `is_night` | Timestamp | 1 if hour ≥ 22 or ≤ 6 |
| `user_id` | Raw | Numerical user identifier |
| `currency` | Raw (encoded) | OrdinalEncoder |
| `location` | Raw (encoded) | OrdinalEncoder — ISO2 country code |
| `merchant` | Raw (encoded) | OrdinalEncoder |

### Fraud Simulation Patterns (Producer)

The producer generates realistic fraud with a target rate of ~1–2%:

| Pattern | Weight | Description |
|---|---|---|
| Account takeover | 40% | High-value txns from compromised user IDs |
| Card testing | 30% | Rapid micro-transactions (< $2) |
| Merchant collusion | 20% | Large amounts at flagged merchants |
| Geographic anomaly | 10% | Impossible location shifts |

### Model Performance (v1 — trained on 332k transactions)

| Metric | Value |
|---|---|
| Precision | 99.2% |
| Recall | 61.3% |
| F1 Score | 75.8% |
| ROC-AUC | 0.829 |
| Decision threshold | 0.9997 |
| Training fraud rate | 0.307% |

---

## Getting Started

### Prerequisites

- Docker Desktop (with at least 8GB RAM allocated)
- A [Confluent Cloud](https://confluent.io) account with a Kafka cluster and `transactions` + `fraud_predictions` topics created
- Python 3.10+ (for local `.venv` only)

### 1. Clone the repository

```bash
git clone <repo-url>
cd fraudlytics/src
```

### 2. Configure environment variables

Copy and populate the secrets file:

```bash
cp .env.example .env
```

Required variables:

```env
FERNET_KEY=<generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
MINIO_USERNAME=minio
MINIO_PASSWORD=minio123
AIRFLOW_UID=50000
KAFKA_BOOTSTRAP_SERVERS=<your-confluent-cluster>:9092
KAFKA_USERNAME=<confluent-api-key>
KAFKA_PASSWORD=<confluent-api-secret>
KAFKA_TOPIC=transactions
```

### 3. Start all services

```bash
docker compose up -d
```

Allow ~2 minutes for all health checks to pass.

### 4. Trigger the first training run

```bash
# Unpause the DAG and trigger a manual run
docker exec src-airflow-worker-1 airflow dags unpause fraud_detection_training
docker exec src-airflow-worker-1 airflow dags trigger fraud_detection_training
```

The first run will:
1. Consume all available transactions from Kafka (~15–30 minutes depending on volume)
2. Train and tune the XGBoost model
3. Register it in MLflow as `fraud_detection_xgboost` with the `champion` alias
4. The inference consumer will automatically reload the new champion

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | airflow / airflow |
| MLflow UI | http://localhost:5500 | — |
| Grafana Dashboard | http://localhost:3001 | admin / admin |
| MinIO Console | http://localhost:9001 | minio / minio123 |

---

## Screenshots

### Grafana — Live Fraud Monitoring Dashboard
<!-- Screenshot: Grafana dashboard showing transaction volume, fraud rate, recent fraud alerts table -->
![Grafana Dashboard](./img/grafana_dashboard.png)

### MLflow — Experiment Tracking & Model Registry
<!-- Screenshot: MLflow experiment run showing metrics, parameters, confusion matrix and ROC curve -->
![MLflow Experiments](./img/mlflow_experiments.png)

### MLflow — Registered Model with Champion Alias
<!-- Screenshot: MLflow model registry showing fraud_detection_xgboost with champion alias -->
![MLflow Model Registry](./img/mlflow_registry.png)

### Airflow — Training DAG
<!-- Screenshot: Airflow DAG graph showing validate_environment → execute_training → promote_model → cleanup_resources all green -->
![Airflow DAG](./img/airflow_dag.png)

### Airflow — Training Task Logs
<!-- Screenshot: execute_training task logs showing Kafka consumption, CV iterations, and final metrics -->
![Airflow Logs](./img/airflow_logs.png)

---

## System Requirements

### Minimum (development)

| Resource | Requirement |
|---|---|
| CPU | 4+ cores (Intel i5/i7 or equivalent) |
| RAM | 16 GB (8 GB allocated to Docker) |
| Storage | 20 GB free |
| GPU | Not required |

### Recommended (production)

| Resource | Requirement |
|---|---|
| CPU | 16 cores (Intel i9 / AMD Ryzen 9) |
| RAM | 64 GB |
| Storage | 1 TB SSD |
| GPU | NVIDIA RTX 3090 (for deep learning extensions) |

### Cloud equivalents

| Provider | Instance |
|---|---|
| AWS | `g5.4xlarge` (GPU) or `m5.4xlarge` (CPU) |
| Azure | `Standard_NC6s_v3` or `Standard_D16s_v3` |
| GCP | `n2-standard-16` or `a2-highgpu-1g` |

---

## Configuration

All tuneable parameters live in `src/config.yaml`:

```yaml
mlflow:
  experiment_name: "fraud_detection"
  registered_model_name: "fraud_detection_xgboost"
  tracking_uri: "http://mlflow-server:5500"
  artifact_location: "s3://mlflow/fraud_detection"
  s3_endpoint_url: "http://minio:9000"
  bucket: "mlflow"

kafka:
  bootstrap_servers: "<confluent-cluster>:9092"
  topic: "transactions"
  output_topic: "fraud_predictions"
```

Key training parameters (in `fraud_detection_training.py`):

| Parameter | Default | Description |
|---|---|---|
| `max_records` | 500,000 | Max Kafka messages consumed per training run |
| `sampling_strategy` | 0.15 | SMOTE target fraud ratio in training set |
| `n_iter` | 15 | RandomizedSearchCV iterations |
| `cv` | 5 | Cross-validation folds |
| `min_recall` | 0.4 | Minimum recall enforced during threshold optimisation |
| `test_size` | 0.2 | Train/test split ratio |
