from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Setup Logging Configuration (at the top)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_monitor")

app = FastAPI(title="Heart Disease API")

# 2. CORS Middleware for Web UI Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

# 4. Load Model
model = joblib.load("models/model.joblib")

# Health Check
@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "healthy"}

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input data to DataFrame for model
        df = pd.DataFrame([data.dict()])
        
        # Perform prediction
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df).max()
        
        label = "Heart Disease" if prediction == 1 else "Normal"
        
        # 5. Integrated Logging for Monitoring
        # This records the event to Docker/Kubernetes logs
        logger.info(f"PREDICTION_EVENT | Input: {data.dict()} | Result: {label} | Confidence: {prob:.2f}")
        
        return {
            "prediction": int(prediction),
            "label": label,
            "confidence": float(prob)
        }
    except Exception as e:
        logger.error(f"PREDICTION_ERROR | Message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))