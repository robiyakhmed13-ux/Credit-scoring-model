# Credit Scoring Model — Fintech ML Project

Predict **loan default risk** from borrower financial data using multiple ML classifiers.

## Why This Matters

Credit scoring is the backbone of fintech — banks, lending platforms, and personal finance apps use it to assess risk, set interest rates, and make lending decisions. This project builds a production-ready scoring pipeline from scratch.

## Results

| Model | Accuracy | F1 | ROC-AUC | CV AUC |
|-------|----------|----|---------|--------|
| **Logistic Regression** | **0.765** | **0.561** | **0.795** | **0.780** ★ |
| Random Forest | 0.756 | 0.508 | 0.786 | 0.757 |
| Gradient Boosting | 0.731 | 0.510 | 0.758 | 0.742 |

**Top predictive features:** interest rate, debt-to-income ratio, number of delinquencies, employment length, credit utilization.

## Pipeline

```
Data Generation (5000 borrowers, 12 raw features)
    → Feature Engineering (+5 derived features: DTI, payment-to-income, risk flags)
    → Preprocessing (StandardScaler, LabelEncoder, stratified 80/20 split)
    → Multi-Model Training (Logistic Regression, Random Forest, Gradient Boosting)
    → Evaluation (accuracy, precision, recall, F1, ROC-AUC, 5-fold CV)
    → Feature Importance Analysis
    → Production Prediction Interface (CreditScorer class with risk grading A-D)
```

## Features

- **17 features** including 5 engineered: debt-to-income, monthly payment ratio, credit line utilization, risk flags
- **3 models compared** with cross-validation
- **Risk grading system**: A (low) → D (very high) with probability scores
- **OOP architecture**: `CreditScorer` class ready for API integration

## Quick Start

```bash
pip install numpy pandas scikit-learn
python credit_scoring_model.py
```

## Tech Stack

Python, scikit-learn, pandas, NumPy, Logistic Regression, Random Forest, Gradient Boosting, StandardScaler, Cross-Validation

## Author

**Robiyakhon Akhmedova** — ML Engineer  
Part of a 50-project ML portfolio covering classification, regression, NLP, CV, and deep learning.
