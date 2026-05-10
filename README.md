# Credit Card Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, with fraud cases representing only 0.17% of all transactions.

The aim is to build a professional fraud detection pipeline covering data validation, exploratory analysis, model training, evaluation, fraud risk scoring, dashboarding, and API deployment.

## Business Problem

Financial institutions process millions of transactions daily. Even a small number of fraudulent transactions can result in major financial losses. This project simulates a real-world fraud analytics workflow by identifying suspicious transactions and assigning fraud risk scores.

## Project Objectives

- Explore transaction patterns between legitimate and fraudulent activity
- Handle severe class imbalance
- Train and compare machine learning models
- Prioritise recall and precision for fraud detection
- Build a fraud risk scoring system
- Deploy results through a dashboard and API

## Exploratory Data Analysis Summary

The dataset contains 284,807 transactions, with only 492 fraud cases. Fraud represents approximately 0.17% of the dataset, making this a highly imbalanced classification problem.

Key findings:
- Fraud cases are rare, so accuracy is not suitable as the main evaluation metric.
- Fraudulent transactions have a higher average transaction amount, but many fraud cases also occur at low values.
- Time-based transaction patterns may support fraud monitoring.
- Several anonymised PCA features show stronger relationships with fraud activity.

Because of the imbalance, model evaluation will focus on recall, precision, F1-score, ROC-AUC, and Precision-Recall AUC.

## Data Preprocessing

The dataset was split into training and testing sets using stratified sampling to preserve the fraud distribution across both datasets.

Feature scaling was applied using StandardScaler on:
- Amount
- Time
- Hour

To prevent data leakage, scaling parameters were learned only from the training data and then applied to the test set.

## Baseline Model Training

Three baseline machine learning models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Because the dataset is highly imbalanced, accuracy was not used as the main performance measure. The main focus was on fraud recall, precision, F1-score, ROC-AUC, and confusion matrix results.

## Model Performance Summary

| Model | Fraud Precision | Fraud Recall | Fraud F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.83 | 0.64 | 0.72 | 0.9574 |
| Random Forest | 0.96 | 0.72 | 0.83 | 0.9530 |
| XGBoost | 0.92 | 0.78 | 0.84 | 0.9485 |

## Key Findings

XGBoost achieved the best fraud recall, correctly identifying 76 out of 98 fraud cases. This makes it the strongest baseline model for detecting fraudulent transactions.

Random Forest achieved the highest fraud precision at 0.96, meaning it produced very few false fraud alerts.

Logistic Regression achieved the highest ROC-AUC score at 0.9574, showing strong overall class separation, but it detected fewer fraud cases compared with Random Forest and XGBoost.

## Business Interpretation

For fraud detection, recall is very important because missing fraudulent transactions can lead to financial loss. Based on the baseline results, XGBoost is currently the preferred model because it detected the highest number of fraud cases while maintaining strong precision.

However, Random Forest may be useful in situations where reducing false alerts is more important.

## Model Improvement: Threshold Tuning

The baseline XGBoost model was improved using class imbalance weighting and threshold tuning.

Because fraud detection is a high-risk business problem, the default classification threshold of 0.5 may not be the best option. A lower threshold can help detect more fraud cases, although it may increase false positives.

Threshold tuning was applied to compare precision, recall, F1-score, false positives, and false negatives at different decision thresholds.

This allows the model to be selected based on business priorities:

- Higher recall reduces missed fraud cases
- Higher precision reduces false fraud alerts
- F1-score balances both precision and recall

## Fraud Risk Scoring System

A fraud risk scoring system was developed using the improved XGBoost model.

Instead of only predicting whether a transaction is fraudulent or legitimate, the model assigns each transaction a fraud probability and converts it into a risk score between 0 and 100.

Risk categories were defined as:

| Risk Score | Risk Category |
|---:|---|
| 0–39 | Low Risk |
| 40–69 | Medium Risk |
| 70–100 | High Risk |

This makes the model output easier for business users, fraud analysts, and risk teams to interpret.

A selected decision threshold of 0.3 was applied to flag suspicious transactions for further review.

## Streamlit Fraud Monitoring Dashboard

An interactive Streamlit dashboard was created to make the fraud detection results easy to interpret.

The dashboard includes:

- Total transaction count
- Actual fraud cases
- Predicted fraud alerts
- High-risk transaction count
- Risk category distribution
- Fraud probability trend
- Transaction risk explorer
- Top 20 highest-risk transactions

This dashboard simulates how fraud analysts or risk teams could monitor suspicious transactions and prioritise cases for review.

## FastAPI Fraud Prediction API

A FastAPI endpoint was created to expose the trained fraud detection model for real-time prediction.

The API accepts transaction features and returns:

- Fraud probability
- Risk score
- Risk category
- Fraud prediction

### Run API

```bash
uvicorn api.main:app --reload
```

### API documentation
```bash
http://127.0.0.1:8000/docs
```

## Model Artifact Notice

The trained model file is not committed to GitHub because model artifacts can be large. To run the API locally, first train the model using:

```bash
python src/threshold_tuning.py
```


