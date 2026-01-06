from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from pydantic import BaseModel
import joblib
import pandas as pd
from .preprocessing import PatientData

app = FastAPI(title="Heart Disease API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Add the instrumentator to the app
Instrumentator().instrument(app).expose(app)

model = joblib.load("models/model.joblib")

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
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df).max()
        
        return {
            "prediction": int(prediction),
            "label": "Heart Disease" if prediction == 1 else "Normal",
            "confidence": float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_monitor")

@app.post("/predict")
def predict(data: HeartData):
    prediction = model.predict(processed_data)
    # Log the result for future drift analysis
    logger.info(f"PREDICTION_LOG: Input={data.dict()}, Result={prediction}")
    return {"prediction": prediction}