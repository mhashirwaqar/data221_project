# Data 221 Project: Credit Card Fraud Detection

## Project Overview
This project focuses on detecting fraudulent credit card transactions using multiple machine learning models on a highly imbalanced dataset (~284,000 transactions).

The objective is to compare different models and determine which performs best for fraud detection, where minimizing false negatives is critical.


## Dataset
- **File:** `creditcard.csv`
- **Size:** ~284,000 rows  

### Features:
- `V1–V28`: PCA-transformed features  
- `Amount`: Transaction amount  
- `Time`: Removed during preprocessing  
- **Target:** `Class`  
  - `0` → Non-Fraud  
  - `1` → Fraud  

The dataset is highly imbalanced (~0.17% fraud cases).


## Data Preprocessing
- Removed irrelevant feature: `Time`
- Train-test split:
  - 80% training / 20% testing  
  - Stratified to preserve class distribution  
- Applied **StandardScaler** for:
  - Logistic Regression  
  - KNN  
  - Neural Network  
- Used **class weighting** to handle class imbalance  


## Models Implemented
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Neural Network (TensorFlow)  


## Evaluation Metrics
Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- Classification Report  


## Model Performance Comparison

| Model                | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|----------|--------|----------|
| Logistic Regression | 0.9745   | 0.0586   | 0.9184 | 0.1101   |
| KNN                 | 0.9996   | 0.9310   | 0.8265 | 0.8757   |
| Decision Tree       | 0.9954   | 0.2413   | 0.7755 | 0.3680   |
| Neural Network      | 0.9602   | 0.0383   | 0.9184 | 0.0735   |

## Analysis
- KNN achieved the best performance with the highest F1-score (0.8757), making it the most effective model for this dataset.
- Logistic Regression and Neural Network achieved high recall but very low precision, resulting in many false positives.
- Decision Tree provided moderate performance but did not outperform KNN.
- The imbalance in the dataset significantly impacts model behavior, especially precision.


## Conclusion
KNN is the most suitable model for this fraud detection task as it provides the best balance between precision and recall.

This project demonstrates the importance of:
- Using appropriate evaluation metrics (F1-score instead of accuracy)
- Handling imbalanced datasets
- Comparing multiple models to identify the best approach


## Tools and Technologies
- Python  
- Scikit-learn  
- TensorFlow / Keras  
- Pandas, NumPy  
- GitHub  


## Group Members
- Muhammad Hashir Waqar  
- Joyce Tran  
- Sean Cheng  
- Mudit Jindal  

Course: DATA 221  
Section: T03  
Date: March 27, 2026  


## License
This project is for academic purposes only.