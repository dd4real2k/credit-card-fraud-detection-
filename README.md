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
