This is a documentation file that serves as the repository landing page (A separate PDF report with screenshots is given)
📝 ## MLOps - Group 60: 
 1. Amod Suresh Puranik (2024aa5507)
 2. Shruthi K (2024aa05806)
 3. Kuna Simha Chalam (2024aa05131)
 4. Lakkavajjala Sowmya (2024aa05317)
 5. Tumuganti Vennela (2024aa05795)

📌 Project Overview
This project demonstrates a production-grade MLOps pipeline for predicting the presence of heart disease. It utilizes the **Heart Disease UCI Dataset** and implements a full lifecycle workflow including:

* **Data Ingestion & Versioning** (UCI Repository)
* **Experiment Tracking** (MLflow)
* **Model Packaging** (Docker)
* **CI/CD Automation** (GitHub Actions)
* **Orchestration & Deployment** (Kubernetes)
* **Monitoring** (Prometheus/Grafana)

## 📂 Repository Structure


heart-disease-mlops/
├── .github/workflows/                      # CI/CD Pipeline definitions
├── api/                                    # FastAPI application & Dockerfile
├── data/                                   # Local data storage (ignored in git)
├── k8s/                                    # Kubernetes manifests (Deployment & Service)
├── mlruns/                                 # Various ML runs
├── models/                                 # Serialized model (joblib)
├── notebooks/                              # Jupyter notebooks for EDA & Experiments
├── src/                                    # Source code for training and processing
├── tests/                                  # Unit and integration tests       
├── Group60-MLOps-Assignment-Report.pdf     # Our Full project report
├── Group60-MLOps-Architecture-diagram.pdf  # Project architecture diagram (large sized)
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation


## 🚀 Getting Started

### 1. Our Tools (full list in requirements.txt file)

* Python 3.11
* Docker Desktop (with Kubernetes enabled)
* MLFLow
* Fast API
* Prometheus + Grafana
* Helm

### 2. Installation

* Install Docker Desktop and WSL (if deploying on local machine)
* Enable Kubernetes in Docker > Settings
* Build the Container Image
* Orchestrate with Helm
* Verify pod health


## 3. Data & Training Pipeline

### Step 1: Data Acquisition & EDA

Download the latest data from the UCI repository and perform EDA:

```bash
python src/data_loader.py
# alternatively EDA Notebook is available at notebooks/01_eda.ipynb

```

### Step 2: Model Training with Experiment Tracking

Train the model. This script automatically logs metrics and artifacts to MLflow.

```bash
python src/train.py

```

* **Artifacts:** The best model is saved to `models/model.joblib`.


## 4. Docker Containerization

### Step 1: Build the API container locally:

```bash
docker build -t heart-disease-api:v4 -f api/Dockerfile .

```

### Step 2: Use Helm to deploy Kubernetes manifests:

```bash
helm install heart-prediction ./charts/heart-disease-api `
  --set image.repository=heart-disease-api `
  --set image.tag=v4 `
  --set image.pullPolicy=IfNotPresent `
  --set replicaCount=1
```

### Step 3: Verify pod health:

```bash
kubectl get pods
kubectl get endpoints heart-service
```

## 4. Access the Service:
The API will be available at `http://localhost:8081` (Swagger UI).

## 5. API Usage

**Endpoint:** `/predict`

**Method:** `POST`

**Sample Request:** Use the following sample JSON input in either the web UI (click "Try it out") or through PowerShell

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0.0,
  "thal": 1.0
}

```

**Sample Response:**

```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 0.85
}

```