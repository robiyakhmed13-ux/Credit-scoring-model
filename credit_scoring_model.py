"""
================================================================================
        CREDIT SCORING MODEL — Fintech ML Project
        Predict loan default risk using borrower financial data
================================================================================

This project builds a credit scoring system that predicts whether a borrower
will default on a loan, using multiple ML classifiers with proper evaluation.

Relevance: Core to fintech operations — banks, lending platforms, and
personal finance apps (like Mylo) use credit scoring to assess risk,
set interest rates, and make lending decisions.

Models Compared:
    1. Logistic Regression (baseline)
    2. Random Forest Classifier
    3. Gradient Boosting Classifier (best performer)

Pipeline:
    1. Synthetic data generation (realistic borrower profiles)
    2. Feature engineering (debt-to-income ratio, risk indicators)
    3. Preprocessing (scaling, encoding)
    4. Multi-model training & comparison
    5. Evaluation (accuracy, precision, recall, F1, ROC-AUC)
    6. Feature importance analysis
    7. Production-ready prediction interface

Author : Robiyakhon Akhmedova
License: MIT
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# 1. DATA GENERATION — Realistic Synthetic Credit Data
# ============================================================================

def generate_credit_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic credit/loan data.

    Features mirror real-world credit bureau data:
        - age, income, employment_length (borrower profile)
        - loan_amount, interest_rate, loan_term (loan terms)
        - credit_history_length, num_credit_lines, num_delinquencies
        - credit_utilization, home_ownership, loan_purpose

    Target: default (0 = repaid, 1 = defaulted)
    Default rate is ~22% (realistic for subprime/mixed portfolios).
    """
    rng = np.random.RandomState(random_state)

    age = rng.randint(21, 65, n_samples)
    income = rng.lognormal(mean=10.5, sigma=0.6, size=n_samples).astype(int)
    income = np.clip(income, 15000, 300000)
    employment_length = rng.randint(0, 30, n_samples)
    loan_amount = (income * rng.uniform(0.1, 0.8, n_samples)).astype(int)
    interest_rate = rng.uniform(3.5, 25.0, n_samples).round(2)
    loan_term = rng.choice([12, 24, 36, 48, 60], n_samples)
    credit_history_length = np.clip(age - 18 - rng.randint(0, 10, n_samples), 0, 45)
    num_credit_lines = rng.randint(1, 20, n_samples)
    num_delinquencies = rng.choice([0]*60 + [1]*20 + [2]*10 + [3]*5 + [4]*3 + [5]*2, n_samples)
    credit_utilization = rng.beta(2, 5, n_samples).round(3)
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], n_samples, p=[0.4, 0.15, 0.45])
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "education", "medical", "business", "auto"],
        n_samples, p=[0.35, 0.15, 0.12, 0.13, 0.10, 0.15]
    )

    # --- Default probability (logistic model with realistic coefficients) ---
    dti = loan_amount / (income + 1)
    log_odds = (
        -3.0
        + 2.5 * dti
        + 0.10 * interest_rate
        + 0.6 * num_delinquencies
        + 2.0 * credit_utilization
        - 0.05 * employment_length
        - 0.04 * credit_history_length
        - 0.02 * (income / 10000)
        + rng.normal(0, 0.4, n_samples)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (rng.random(n_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "employment_length": employment_length,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "loan_term": loan_term,
        "credit_history_length": credit_history_length,
        "num_credit_lines": num_credit_lines,
        "num_delinquencies": num_delinquencies,
        "credit_utilization": credit_utilization,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "default": default,
    })
    return df


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that capture financial risk signals.

    New features:
        - debt_to_income    : loan_amount / income (key risk ratio)
        - monthly_payment   : estimated monthly installment
        - payment_to_income : monthly payment as % of monthly income
        - credit_line_util  : utilization per credit line
        - risk_flag         : high delinquency + high utilization
    """
    df = df.copy()
    df["debt_to_income"] = (df["loan_amount"] / df["income"]).round(4)
    df["monthly_payment"] = (df["loan_amount"] / df["loan_term"]).round(2)
    df["payment_to_income"] = (df["monthly_payment"] / (df["income"] / 12)).round(4)
    df["credit_line_util"] = (df["credit_utilization"] / (df["num_credit_lines"] + 1)).round(4)
    df["risk_flag"] = ((df["num_delinquencies"] >= 2) & (df["credit_utilization"] > 0.5)).astype(int)
    return df


# ============================================================================
# 3. PREPROCESSING
# ============================================================================

def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, scale numerics, split into train/test.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = df.copy()

    # Label-encode categoricals
    le_home = LabelEncoder()
    le_purpose = LabelEncoder()
    df["home_ownership"] = le_home.fit_transform(df["home_ownership"])
    df["loan_purpose"] = le_purpose.fit_transform(df["loan_purpose"])

    # Separate features / target
    X = df.drop("default", axis=1)
    y = df["default"]
    feature_names = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"  Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}  |  Default rate: {y.mean():.1%}")
    return X_train, X_test, y_train, y_test, scaler, feature_names


# ============================================================================
# 4. MODEL TRAINING & COMPARISON
# ============================================================================

def train_and_compare(X_train, X_test, y_train, y_test, feature_names):
    """
    Train three classifiers, evaluate each, and return results.

    Models:
        - Logistic Regression  (fast, interpretable baseline)
        - Random Forest        (handles non-linearity, feature importance)
        - Gradient Boosting    (best accuracy on tabular data)
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    best_model = None
    best_auc = 0

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": auc, "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(), "model": model
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}  |  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        if auc > best_auc:
            best_auc = auc
            best_model = name

    print(f"\n  ★ Best model: {best_model} (ROC-AUC: {best_auc:.4f})")

    # Feature importance from the best tree-based model
    gb_model = results["Gradient Boosting"]["model"]
    importances = pd.Series(gb_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print(f"\n  Top 5 Features (Gradient Boosting):")
    for feat, imp in importances.head(5).items():
        bar = "█" * int(imp * 60)
        print(f"    {feat:25s} {imp:.4f}  {bar}")

    return results, best_model


# ============================================================================
# 5. PREDICTION INTERFACE
# ============================================================================

class CreditScorer:
    """
    Production-ready credit scoring interface.

    Usage:
        scorer = CreditScorer(model, scaler, feature_names)
        risk = scorer.score_borrower({
            "age": 35, "income": 55000, ...
        })
    """

    RISK_BANDS = [
        (0.0, 0.15, "LOW RISK",    "A"),
        (0.15, 0.35, "MEDIUM RISK", "B"),
        (0.35, 0.55, "HIGH RISK",   "C"),
        (0.55, 1.01, "VERY HIGH",   "D"),
    ]

    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def score_borrower(self, borrower: dict) -> dict:
        """Score a single borrower and return risk assessment."""

        # Engineer features
        b = borrower.copy()
        b["debt_to_income"] = round(b["loan_amount"] / b["income"], 4)
        b["monthly_payment"] = round(b["loan_amount"] / b["loan_term"], 2)
        b["payment_to_income"] = round(b["monthly_payment"] / (b["income"] / 12), 4)
        b["credit_line_util"] = round(b["credit_utilization"] / (b["num_credit_lines"] + 1), 4)
        b["risk_flag"] = int(b["num_delinquencies"] >= 2 and b["credit_utilization"] > 0.5)

        # Encode categoricals (simplified for demo)
        home_map = {"MORTGAGE": 0, "OWN": 1, "RENT": 2}
        purpose_map = {"auto": 0, "business": 1, "debt_consolidation": 2,
                       "education": 3, "home_improvement": 4, "medical": 5}
        b["home_ownership"] = home_map.get(b["home_ownership"], 2)
        b["loan_purpose"] = purpose_map.get(b["loan_purpose"], 2)

        # Build feature vector
        X = np.array([[b[f] for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)

        # Predict
        prob_default = self.model.predict_proba(X_scaled)[0][1]
        prediction = int(prob_default >= 0.5)

        # Risk band
        for lo, hi, label, grade in self.RISK_BANDS:
            if lo <= prob_default < hi:
                risk_label, risk_grade = label, grade
                break

        return {
            "default_probability": round(prob_default, 4),
            "prediction": "DEFAULT" if prediction else "REPAID",
            "risk_grade": risk_grade,
            "risk_label": risk_label,
        }


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  CREDIT SCORING MODEL — Fintech ML Project")
    print("  Predict loan default risk from borrower financial data")
    print("=" * 70)

    # --- Data ---
    print("\n▸ STEP 1: Generating credit data...")
    df = generate_credit_data(n_samples=5000)
    print(f"  Generated {len(df)} borrower profiles")
    print(f"  Default rate: {df['default'].mean():.1%}")

    # --- Feature engineering ---
    print("\n▸ STEP 2: Feature engineering...")
    df = engineer_features(df)
    print(f"  Added 5 derived features → {df.shape[1] - 1} total features")

    # --- Preprocessing ---
    print("\n▸ STEP 3: Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # --- Training ---
    print("\n▸ STEP 4: Training & comparing models...")
    results, best_name = train_and_compare(X_train, X_test, y_train, y_test, feature_names)

    # --- Detailed report on best model ---
    best = results[best_name]
    y_pred = best["model"].predict(X_test)
    print(f"\n{'=' * 70}")
    print(f"  BEST MODEL REPORT: {best_name}")
    print(f"{'=' * 70}")
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Repaid", "Default"])
    for line in report.split("\n"):
        print(f"    {line}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                  Predicted Repaid  Predicted Default")
    print(f"    Actual Repaid      {cm[0][0]:5d}          {cm[0][1]:5d}")
    print(f"    Actual Default     {cm[1][0]:5d}          {cm[1][1]:5d}")

    # --- Prediction demo ---
    print(f"\n{'=' * 70}")
    print("  PREDICTION DEMO")
    print(f"{'=' * 70}")

    scorer = CreditScorer(best["model"], scaler, feature_names)

    demo_borrowers = [
        {
            "label": "Low-risk borrower (high income, no delinquencies)",
            "data": {
                "age": 42, "income": 95000, "employment_length": 15,
                "loan_amount": 20000, "interest_rate": 6.5, "loan_term": 36,
                "credit_history_length": 18, "num_credit_lines": 8,
                "num_delinquencies": 0, "credit_utilization": 0.15,
                "home_ownership": "OWN", "loan_purpose": "home_improvement",
            }
        },
        {
            "label": "High-risk borrower (low income, multiple delinquencies)",
            "data": {
                "age": 25, "income": 28000, "employment_length": 1,
                "loan_amount": 22000, "interest_rate": 22.0, "loan_term": 60,
                "credit_history_length": 2, "num_credit_lines": 5,
                "num_delinquencies": 3, "credit_utilization": 0.72,
                "home_ownership": "RENT", "loan_purpose": "debt_consolidation",
            }
        },
    ]

    for borrower in demo_borrowers:
        result = scorer.score_borrower(borrower["data"])
        print(f"\n  👤 {borrower['label']}")
        print(f"     Default Probability: {result['default_probability']:.1%}")
        print(f"     Prediction:          {result['prediction']}")
        print(f"     Risk Grade:          {result['risk_grade']} ({result['risk_label']})")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<25} {'Accuracy':>9} {'F1':>7} {'ROC-AUC':>9} {'CV AUC':>9}")
    print(f"  {'-'*60}")
    for name, r in results.items():
        star = " ★" if name == best_name else ""
        print(f"  {name:<25} {r['accuracy']:>8.4f} {r['f1']:>7.4f} {r['roc_auc']:>8.4f} {r['cv_auc_mean']:>8.4f}{star}")

    print(f"\n  ✓ Pipeline complete. Model ready for deployment.\n")


if __name__ == "__main__":
    main()
