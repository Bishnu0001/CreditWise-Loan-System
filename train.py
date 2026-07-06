"""
CreditWise Loan System - Model Training Script
------------------------------------------------
Builds a full preprocessing + classification pipeline (imputation, encoding,
scaling, model) and saves it to model.pkl so the Streamlit app can load it
instantly without retraining.

Run once locally before deploying:
    python train.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)

DATA_PATH = "loan_approval_data.csv"
MODEL_PATH = "model.pkl"

TARGET = "Loan_Approved"
DROP_COLS = ["Applicant_ID"]

CATEGORICAL_COLS = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Education_Level", "Gender", "Employer_Category",
]

NUMERIC_COLS = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
    "Collateral_Value", "Loan_Amount", "Loan_Term",
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop rows where the target itself is missing (can't train/evaluate on these)
    df = df.dropna(subset=[TARGET])

    # Normalise target to 1/0
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    df = df.drop(columns=DROP_COLS)
    return df


def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])

    return preprocessor


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---")
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:   ", round(recall_score(y_test, y_pred), 3))
    print("F1 score: ", round(f1_score(y_test, y_pred), 3))
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return f1_score(y_test, y_pred)


def main():
    df = load_data()

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }

    best_name, best_pipeline, best_score = None, None, -1

    for name, clf in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)
        score = evaluate(name, pipeline, X_test, y_test)
        if score > best_score:
            best_name, best_pipeline, best_score = name, pipeline, score

    print(f"\nBest model selected (by F1 score): {best_name}")

    # Refit best pipeline on the FULL dataset before shipping it
    final_pipeline = best_pipeline
    final_pipeline.fit(X, y)

    joblib.dump({
        "pipeline": final_pipeline,
        "model_name": best_name,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }, MODEL_PATH)

    print(f"\nSaved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
