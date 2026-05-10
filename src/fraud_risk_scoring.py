import pandas as pd
import joblib


# Load improved model and test data
model = joblib.load("../models/improved_xgboost_model.pkl")

X_test = pd.read_csv("../data/processed/X_test.csv")
y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()


# Predict fraud probability
fraud_probability = model.predict_proba(X_test)[:, 1]


# Convert probability to risk score from 0 to 100
risk_score = fraud_probability * 100


# Assign risk category
def assign_risk_category(score):
    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"


results = X_test.copy()
results["Actual_Class"] = y_test
results["Fraud_Probability"] = fraud_probability
results["Risk_Score"] = risk_score
results["Risk_Category"] = results["Risk_Score"].apply(assign_risk_category)


# Business decision using selected threshold
selected_threshold = 0.3
results["Predicted_Fraud"] = (results["Fraud_Probability"] >= selected_threshold).astype(int)


# Save output
results.to_csv("../reports/fraud_risk_scored_transactions.csv", index=False)

print("Fraud risk scoring completed successfully.")
print(results[["Actual_Class", "Fraud_Probability", "Risk_Score", "Risk_Category", "Predicted_Fraud"]].head(10))
print(results["Risk_Category"].value_counts())
