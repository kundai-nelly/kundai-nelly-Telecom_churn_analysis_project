"""
train_model.py — Customer churn prediction, Telco dataset.

Cleans the raw WA_Fn-UseC_-Telco-Customer-Churn.csv, runs the preprocessing
pipeline, trains and compares three candidate models, and saves the selected
production model to churn_production_model.pkl.

This is the script form of notebooks/Customer_Churn_Analysis.ipynb — same
logic, meant to be run non-interactively (e.g. to retrain on refreshed data).

Run with:  python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = HERE / "churn_production_model.pkl"

NO_SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw export and fix the two data-entry issues in the source column.

    TotalCharges is exported as text; 11 rows are blank (all customers with
    tenure == 0, i.e. brand-new sign-ups with no bill yet). Coercing to numeric
    turns those into NaN, and we drop them rather than impute a bill that
    doesn't exist yet.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"Dropped {n_dropped} rows with blank TotalCharges (new sign-ups, tenure=0).")

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    for col in NO_SERVICE_COLS:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )
    df = df.drop(columns=["customerID"])
    return df


def build_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def evaluate(name, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def main():
    df = load_and_clean()
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        metrics = evaluate(name, pipeline, X_test, y_test)
        results.append(metrics)
        print(f"\n{name}")
        print(f"  accuracy={metrics['accuracy']:.3f}  precision={metrics['precision']:.3f}  "
              f"recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
        print(f"  confusion matrix: {metrics['confusion_matrix']}")

    # Production choice: Logistic Regression (balanced). It isn't just the
    # interpretable option here, it's also the best one on the metrics that
    # matter for churn: recall 0.80 vs 0.57 (plain) and 0.48 (Random Forest,
    # balanced) — class_weight='balanced' pushes trees to rebalance splits,
    # not the decision threshold, so it doesn't buy the same recall gain it
    # does for a linear model. F1 (0.608) and ROC-AUC (0.835, effectively
    # tied with plain logistic's 0.836) both hold up. A missed churner costs
    # more than a wasted retention call, so the recall gain is worth the
    # precision it gives up (0.49 vs 0.65) — and the coefficients still let a
    # retention rep see why a customer scored high risk.
    chosen_name = "Logistic Regression (balanced)"
    chosen_pipeline = fitted[chosen_name]
    chosen_metrics = [r for r in results if r["name"] == chosen_name][0]

    threshold = 0.5
    artifact = {
        "pipeline": chosen_pipeline,
        "threshold": threshold,
        "model_name": chosen_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": chosen_metrics,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved {chosen_name} to {MODEL_PATH}")

    return results, chosen_metrics


if __name__ == "__main__":
    main()
