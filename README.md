# Credit Card Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, with fraud cases representing only 0.17% of all transactions.

The aim is to build a professional fraud detection pipeline covering data validation, exploratory analysis, model training, evaluation, fraud risk scoring, dashboarding, and API deployment.

## Dataset

Dataset source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymised European credit card transactions made over two days in September 2013.

## Business Problem

Financial institutions process millions of transactions daily. Even a small number of fraudulent transactions can result in major financial losses. This project simulates a real-world fraud analytics workflow by identifying suspicious transactions and assigning fraud risk scores.

## Project Objectives

- Explore transaction patterns between legitimate and fraudulent activity
- Handle severe class imbalance
- Train and compare machine learning models
- Prioritise recall and precision for fraud detection
- Build a fraud risk scoring system
- Deploy results through a dashboard and API

## Project Architecture

```text
credit-card-fraud-detection/
├── api/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Model Performance Summary

| Model | Fraud Precision | Fraud Recall | Fraud F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.83 | 0.64 | 0.72 | 0.9574 |
| Random Forest | 0.96 | 0.72 | 0.83 | 0.9530 |
| XGBoost | 0.92 | 0.78 | 0.84 | 0.9485 |

## Threshold Tuning

The improved XGBoost model was evaluated across several decision thresholds. This is important because the default threshold of 0.5 may not be suitable for fraud detection.

| Threshold | Precision | Recall | F1-score | False Positives | False Negatives | Fraud Detected |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.177 | 0.908 | 0.296 | 415 | 9 | 89 |
| 0.2 | 0.297 | 0.888 | 0.445 | 206 | 11 | 87 |
| 0.3 | 0.424 | 0.888 | 0.574 | 118 | 11 | 87 |
| 0.4 | 0.527 | 0.888 | 0.662 | 78 | 11 | 87 |
| 0.5 | 0.584 | 0.888 | 0.704 | 62 | 11 | 87 |

A selected threshold of 0.3 was used for the risk scoring system to prioritise fraud detection while keeping alerts manageable.

## Fraud Risk Scoring System

The model converts fraud probability into a business-friendly risk score between 0 and 100.

| Risk Score | Risk Category |
|---:|---|
| 0–39 | Low Risk |
| 40–69 | Medium Risk |
| 70–100 | High Risk |

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/dd4real2k/credit-card-fraud-detection-.git
cd credit-card-fraud-detection-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download `creditcard.csv` from Kaggle and place it in:

```text
data/raw/creditcard.csv
```

### 4. Run preprocessing

```bash
python src/data_preprocessing.py
```

### 5. Train baseline models

```bash
python src/train_model.py
```

### 6. Run threshold tuning

```bash
python src/threshold_tuning.py
```

### 7. Generate fraud risk scores

```bash
python src/fraud_risk_scoring.py
```

### 8. Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

### 9. Run the FastAPI server

```bash
uvicorn api.main:app --reload
```

Then open the API documentation at `/docs`.

## Recruiter Summary

This project demonstrates an end-to-end fraud detection system using Python, machine learning, risk scoring, dashboarding, API development, and Docker deployment.

The project is relevant to roles in data analytics, fraud analytics, risk monitoring, fintech, cybersecurity analytics, and machine learning operations.

## Key Skills Demonstrated

- Python data analysis
- Machine learning classification
- Imbalanced dataset handling
- Fraud analytics
- Risk scoring
- Model evaluation
- Streamlit dashboard development
- FastAPI API development
- Docker deployment
- Git and GitHub version control
- Business problem solving

## Future Improvements

- Add automated model retraining pipeline
- Deploy API to a cloud platform
- Add PostgreSQL for transaction storage
- Add real-time transaction streaming
- Build authentication for API users
- Add monitoring for model drift
- Improve explainability using SHAP
