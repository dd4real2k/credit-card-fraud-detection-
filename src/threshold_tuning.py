import pandas as pd
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier


# Load data
X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()


# Train improved XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)


# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]


# Test different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

results = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positives": cm[0][1],
        "false_negatives": cm[1][0],
        "true_frauds_detected": cm[1][1]
    })


results_df = pd.DataFrame(results)
print(results_df)


# Save model and results
joblib.dump(model, "../models/improved_xgboost_model.pkl")
results_df.to_csv("../reports/threshold_tuning_results.csv", index=False)

print("Improved model and threshold results saved successfully.")
