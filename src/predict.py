import joblib
import pandas as pd


MODEL_PATH = "../models/improved_xgboost_model.pkl"
SAMPLE_DATA_PATH = "../data/processed/X_test.csv"


def assign_risk_category(score):
    """Convert fraud risk score into a business-friendly risk category."""

    if score >= 70:
        return "High Risk"
    if score >= 40:
        return "Medium Risk"
    return "Low Risk"


def predict_single_transaction(transaction, threshold=0.3):
    """Predict fraud risk for a single transaction."""

    model = joblib.load(MODEL_PATH)

    transaction_df = pd.DataFrame([transaction])

    fraud_probability = model.predict_proba(transaction_df)[0][1]
    risk_score = fraud_probability * 100
    risk_category = assign_risk_category(risk_score)
    predicted_fraud = int(fraud_probability >= threshold)

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_score": round(float(risk_score), 2),
        "risk_category": risk_category,
        "predicted_fraud": predicted_fraud,
    }


if __name__ == "__main__":
    sample_data = pd.read_csv(SAMPLE_DATA_PATH)

    sample_transaction = sample_data.iloc[0].to_dict()
    prediction = predict_single_transaction(sample_transaction)

    print("\n========== Sample Fraud Prediction ==========\n")
    print(prediction)
