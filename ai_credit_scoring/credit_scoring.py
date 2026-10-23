"""
AI Credit Scoring - simple base project
---------------------------------------
Pipeline:
  1. Generate (or load) applicant data
  2. Preprocess + split
  3. Train Logistic Regression and Random Forest
  4. Evaluate (accuracy, ROC-AUC, confusion matrix)
  5. Convert default probability -> credit score (300-850)
  6. Score a new applicant

Install:  pip install numpy pandas scikit-learn matplotlib joblib
Run:      python credit_scoring.py
To use real data: replace load_data() with pd.read_csv("your.csv")
and make sure it has a binary 'default' column (1 = defaulted).
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

RANDOM_STATE = 42
FEATURES = ["age", "income", "loan_amount", "credit_history_years",
            "num_open_accounts", "late_payments", "debt_to_income",
            "credit_utilization"]


# ------------------------------------------------------------------ 1. DATA
def load_data(n=5000):
    """Create synthetic applicant data with a realistic default pattern."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "age": rng.integers(21, 70, n),
        "income": rng.normal(50000, 18000, n).clip(12000),
        "loan_amount": rng.normal(15000, 8000, n).clip(1000),
        "credit_history_years": rng.integers(0, 30, n),
        "num_open_accounts": rng.integers(1, 15, n),
        "late_payments": rng.poisson(1.2, n),
        "credit_utilization": rng.beta(2, 4, n),
    })
    df["debt_to_income"] = (df["loan_amount"] / df["income"]).round(3)

    # Hidden rule that generates defaults (the model must learn this)
    z = (-2.5
         + 0.45 * df["late_payments"]
         + 2.5 * df["debt_to_income"]
         + 2.0 * df["credit_utilization"]
         - 0.04 * df["credit_history_years"]
         - 0.000015 * df["income"]
         + rng.normal(0, 0.5, n))
    prob = 1 / (1 + np.exp(-z))
    df["default"] = (rng.random(n) < prob).astype(int)
    return df


# ------------------------------------------------------------ 2. MODELS
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1),
    }


# ------------------------------------------------------------ 3. SCORE
def prob_to_score(p_default, base_score=600, base_odds=10, pdo=20):
    """
    Standard scorecard scaling: score = offset + factor * ln(odds good:bad)
    base_score at base_odds (good:bad); every `pdo` points doubles the odds.
    Result is clipped to the familiar 300-850 range.
    """
    p = np.clip(p_default, 1e-6, 1 - 1e-6)
    odds = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return np.clip(offset + factor * np.log(odds), 300, 850).round().astype(int)


def risk_band(score):
    if score >= 750: return "Excellent - Approve"
    if score >= 680: return "Good - Approve"
    if score >= 600: return "Fair - Manual review"
    if score >= 520: return "Poor - High risk"
    return "Very poor - Reject"


# ------------------------------------------------------------- MAIN
def main():
    df = load_data()
    print(f"Dataset: {df.shape[0]} rows | default rate: {df['default'].mean():.1%}\n")

    X, y = df[FEATURES], df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results, plt_data = {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.3).astype(int)  # lower threshold: catch more defaulters
        results[name] = {"model": model,
                         "acc": accuracy_score(y_test, pred),
                         "auc": roc_auc_score(y_test, proba)}
        plt_data[name] = roc_curve(y_test, proba)

        print(f"=== {name} ===")
        print(f"Accuracy: {results[name]['acc']:.3f} | ROC-AUC: {results[name]['auc']:.3f}")
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, target_names=["Good", "Default"]))

    # Pick best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]["model"]
    print(f"Best model: {best_name}\n")

    # ROC plot
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, _) in plt_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=120)

    # Feature importance (Random Forest)
    rf = results["Random Forest"]["model"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(6, 4))
    imp.plot(kind="barh"); plt.title("Feature Importance (Random Forest)")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=120)

    # Save model
    joblib.dump(best, "credit_model.joblib")

    # Score a new applicant
    applicant = pd.DataFrame([{
        "age": 34, "income": 65000, "loan_amount": 12000,
        "credit_history_years": 8, "num_open_accounts": 5,
        "late_payments": 1, "debt_to_income": 12000 / 65000,
        "credit_utilization": 0.25,
    }])
    p_def = best.predict_proba(applicant[FEATURES])[0, 1]
    score = int(prob_to_score(p_def))
    print("--- New applicant ---")
    print(applicant.iloc[0].to_string())
    print(f"\nProbability of default: {p_def:.1%}")
    print(f"Credit score: {score}  ->  {risk_band(score)}")


if __name__ == "__main__":
    main()
