import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


# Load datasets
X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")

y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()


# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    ),
}


best_model = None
best_f1_score = 0


# Train and evaluate
for name, model in models.items():
    print(f"\n========== {name} ==========\n")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    fraud_f1 = f1_score(y_test, predictions)
    roc_score = roc_auc_score(y_test, probabilities)

    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print(f"Fraud F1-score: {fraud_f1:.4f}")
    print(f"ROC-AUC Score: {roc_score:.4f}")

    if fraud_f1 > best_f1_score:
        best_f1_score = fraud_f1
        best_model = model


# Save best model
joblib.dump(best_model, "../models/fraud_detection_model.pkl")

print("\nBest model saved successfully.")
print(f"Best fraud F1-score: {best_f1_score:.4f}")
