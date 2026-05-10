import joblib
import pandas as pd


MODEL_PATH = "../models/improved_xgboost_model.pkl"
X_TEST_PATH = "../data/processed/X_test.csv"
Y_TEST_PATH = "../data/processed/y_test.csv"
OUTPUT_PATH = "../reports/fraud_risk_scored_transactions.csv"

SELECTED_THRESHOLD = 0.3


def assign_risk_category(score):
    """Convert risk score into a business-friendly risk category."""

    if score >= 70:
        return "High Risk"
    if score >= 40:
        return "Medium Risk"
    return "Low Risk"


def generate_risk_scores():
    """Generate fraud probabilities, risk scores, and risk categories."""

    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    fraud_probabilities = model.predict_proba(X_test)[:, 1]
    risk_scores = fraud_probabilities * 100

    results = X_test.copy()
    results["Actual_Class"] = y_test
    results["Fraud_Probability"] = fraud_probabilities
    results["Risk_Score"] = risk_scores
    results["Risk_Category"] = results["Risk_Score"].apply(assign_risk_category)
    results["Predicted_Fraud"] = (
        results["Fraud_Probability"] >= SELECTED_THRESHOLD
    ).astype(int)

    return results


def main():
    """Create and save fraud risk scored transactions."""

    results = generate_risk_scores()
    results.to_csv(OUTPUT_PATH, index=False)

    print("Fraud risk scoring completed successfully.")
    print(results["Risk_Category"].value_counts())
    print(
        results[
            [
                "Actual_Class",
                "Fraud_Probability",
                "Risk_Score",
                "Risk_Category",
                "Predicted_Fraud",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
