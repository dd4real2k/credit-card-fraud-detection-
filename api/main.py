from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "improved_xgboost_model.pkl"
SELECTED_THRESHOLD = 0.3


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting credit card fraud risk using a trained XGBoost model.",
    version="1.0.0",
)


model = joblib.load(MODEL_PATH)


class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Hour: float


def assign_risk_category(score):
    """Convert fraud risk score into a business-friendly risk category."""

    if score >= 70:
        return "High Risk"
    if score >= 40:
        return "Medium Risk"
    return "Low Risk"


@app.get("/")
def home():
    """API health check endpoint."""

    return {
        "message": "Credit Card Fraud Detection API is running",
        "docs": "/docs",
    }


@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    """Predict fraud probability, risk score, and risk category."""

    input_data = pd.DataFrame([transaction.model_dump()])

    fraud_probability = model.predict_proba(input_data)[0][1]
    risk_score = fraud_probability * 100
    risk_category = assign_risk_category(risk_score)
    predicted_fraud = int(fraud_probability >= SELECTED_THRESHOLD)

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_score": round(float(risk_score), 2),
        "risk_category": risk_category,
        "predicted_fraud": predicted_fraud,
    }
