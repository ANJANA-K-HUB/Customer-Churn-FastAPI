import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API service powered by FastAPI and Scikit-Learn.",
    version="1.0.0",
)

# 2. Load Trained Model
try:
    model = joblib.load("churn_model.pkl")
except Exception:
    model = None


# 3. Define Input Data Schema with Pydantic Validation
class CustomerInput(BaseModel):
    tenure: int = Field(
        ..., ge=1, le=120, description="Months with company", example=12
    )
    monthly_charges: float = Field(
        ..., gt=0, description="Monthly bill in USD", example=85.50
    )
    total_charges: float = Field(
        ..., gt=0, description="Total spend in USD", example=1026.00
    )


# 4. Define Output Response Schema
class PredictionOutput(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


# 5. Root & Health Check Endpoints
@app.get("/")
def home():
    return {
        "message": "Welcome to Customer Churn Prediction API! Visit /docs for interactive testing."
    }


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(
            status_code=500, detail="ML Model artifact is missing."
        )
    return {"status": "healthy", "model_loaded": True}


# 6. Model Prediction Endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model artifact not loaded."
        )

    # Format input data for Scikit-Learn model (2D Array)
    features = np.array(
        [[data.tenure, data.monthly_charges, data.total_charges]]
    )

    # Perform Inference
    pred_class = int(model.predict(features)[0])
    pred_prob = float(model.predict_proba(features)[0][1])

    risk = "High Risk" if pred_class == 1 else "Low Risk"

    return PredictionOutput(
        churn_prediction=pred_class,
        churn_probability=round(pred_prob * 100, 2),
        risk_level=risk,
    )