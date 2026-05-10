from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train_and_evaluate_models():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    results = []
    best_model = None
    best_model_name = None
    best_f1_score = 0

    for name, model in models.items():
        print(f"\n========== {name} ==========")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, predictions)
        fraud_f1 = f1_score(y_test, predictions)

        print(classification_report(y_test, predictions))
        print("Confusion Matrix:")
        print(cm)

        results.append(
            {
                "model": name,
                "fraud_f1_score": fraud_f1,
                "roc_auc": roc_auc_score(y_test, probabilities),
                "pr_auc": average_precision_score(y_test, probabilities),
                "false_positives": int(cm[0][1]),
                "false_negatives": int(cm[1][0]),
                "true_frauds_detected": int(cm[1][1]),
            }
        )

        if fraud_f1 > best_f1_score:
            best_f1_score = fraud_f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results).sort_values(by="fraud_f1_score", ascending=False)
    results_df.to_csv(REPORTS_DIR / "baseline_model_results.csv", index=False)
    joblib.dump(best_model, MODEL_DIR / "fraud_detection_model.pkl")

    print("\nBest baseline model saved successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Best fraud F1-score: {best_f1_score:.4f}")


if __name__ == "__main__":
    train_and_evaluate_models()
