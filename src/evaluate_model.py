import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


MODEL_PATH = "../models/improved_xgboost_model.pkl"
X_TEST_PATH = "../data/processed/X_test.csv"
Y_TEST_PATH = "../data/processed/y_test.csv"


def evaluate_model(threshold=0.3):
    """Evaluate the trained fraud detection model using key classification metrics."""

    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    fraud_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (fraud_probabilities >= threshold).astype(int)

    print("\n========== Fraud Detection Model Evaluation ==========\n")
    print(f"Selected threshold: {threshold}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nKey Metrics:")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall: {recall_score(y_test, predictions):.4f}")
    print(f"F1-score: {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, fraud_probabilities):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, fraud_probabilities):.4f}")


if __name__ == "__main__":
    evaluate_model()
