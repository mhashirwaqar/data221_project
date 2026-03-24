# Data 221 Project: Credit Card Fraud Detection


## Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning models on a highly imbalanced dataset (~284,000 rows).



## Team Roles & Responsibilities

- **Member 1**
  - Data preprocessing (cleaning, train/test split)
  - Logistic Regression model

- **Member 2**
  - Feature scaling
  - KNN model + hyperparameter tuning

- **Member 3**
  - Visualization (confusion matrix, plots)
  - Decision Tree model

- **Member 4**
  - Evaluation (metrics collection, comparison table)
  - Neural Network (MLP)


## Project Setup

- Methodology:
  - Class weighting

- Scaling:
  - StandardScaler

- Tools:
  - Python (pandas, sklearn)
  - GitHub (collaboration)
  - Communication (Discord/ Email)

---

## Dataset

- File: `creditcard.csv`
- Size: ~284,000 rows
- Features:
  - V1–V28 (PCA transformed)
  - Time, Amount
  - Target: `Class` (fraud vs non-fraud)

---

## Data Pipeline

- Load dataset using pandas
- Explore data (shape, class distribution)
- Train-test split (80/20 with stratification)
- Apply scaling where needed
- Handle imbalance using class weights

---

## Models Implemented

- Logistic Regression  

- K-Nearest Neighbors (KNN)  

- Decision Tree  

- Neural Network (MLP)  

---

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Analysis

- Compare model performance
- Discuss precision vs recall trade-offs
- Explain impact of class imbalance
- Identify best-performing model

