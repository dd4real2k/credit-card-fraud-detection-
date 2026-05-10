import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


X_TRAIN_PATH = "../data/processed/X_train.csv"
X_TEST_PATH = "../data/processed/X_test.csv"
Y_TRAIN_PATH = "../data/processed/y_train.csv"
Y_TEST_PATH = "../data/processed/y_test.csv"

MODEL_OUTPUT_PATH = "../models/improved_xgboost_model.pkl"
THRESHOLD_RESULTS_PATH = "../reports/threshold_tuning_results.csv"


def load_data():
    """Load preprocessed train and test datasets."""

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return X_train, X_test, y_train, y_test


def train_improved_xgboost(X_train, y_train):
    """Train XGBoost model with class imbalance weighting."""

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_thresholds(y_test, fraud_probabilities):
    """Evaluate model predictions at multiple decision thresholds."""

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for threshold in thresholds:
        predictions = (fraud_probabilities >= threshold).astype(int)

        cm = confusion_matrix(y_test, predictions)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions),
                "false_positives": cm[0][1],
                "false_negatives": cm[1][0],
                "true_frauds_detected": cm[1][1],
            }
        )

    return pd.DataFrame(results)


def main():
    """Train improved model, evaluate thresholds, and save outputs."""

    X_train, X_test, y_train, y_test = load_data()

    model = train_improved_xgboost(X_train, y_train)

    fraud_probabilities = model.predict_proba(X_test)[:, 1]
    results_df = evaluate_thresholds(y_test, fraud_probabilities)

    print(results_df)

    joblib.dump(model, MODEL_OUTPUT_PATH)
    results_df.to_csv(THRESHOLD_RESULTS_PATH, index=False)

    print("Improved model and threshold tuning results saved successfully.")


if __name__ == "__main__":
    main()
