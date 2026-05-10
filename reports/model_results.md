# Baseline Model Results

## Logistic Regression

- Fraud Precision: 0.83
- Fraud Recall: 0.64
- Fraud F1-score: 0.72
- ROC-AUC: 0.9574
- False Negatives: 35
- False Positives: 13

## Random Forest

- Fraud Precision: 0.96
- Fraud Recall: 0.72
- Fraud F1-score: 0.83
- ROC-AUC: 0.9530
- False Negatives: 27
- False Positives: 3

## XGBoost

- Fraud Precision: 0.92
- Fraud Recall: 0.78
- Fraud F1-score: 0.84
- ROC-AUC: 0.9485
- False Negatives: 22
- False Positives: 7

## Final Baseline Decision

XGBoost is selected as the best baseline model because it achieved the highest fraud recall and F1-score. In fraud detection, reducing false negatives is especially important because missed fraud cases may result in financial loss.
