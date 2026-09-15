from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
REPORT_DIR = Path("reports")
METRICS_PATH = REPORT_DIR / "churn_model_metrics.json"
CM_PATH = REPORT_DIR / "churn_confusion_matrix.png"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """Load the raw dataset and clean the relevant churn columns."""
    df = pd.read_csv(path)

    # Normalize the target and make labels binary for the classifier.
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Convert TotalCharges to numeric, preserving missings instead of crashing.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customer ID: unique identifier, not a predictive feature.
    df = df.drop(columns=["customerID"], errors="ignore")

    # Remove rows with missing target before train/test.
    df = df.dropna(subset=["Churn"]).reset_index(drop=True)

    return df


def build_pipeline() -> Pipeline:
    """Create a preprocessing + estimator pipeline for the churn model."""
    # Numeric preprocessing: median imputation and scaling.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing: most-frequent imputation and one-hot encoding.
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    estimator = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Tune the model with a small search grid for C values only."""
    params = {
        "model__C": [0.1, 1, 5, 10],
    }

    grid = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=params,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid


def select_threshold(clf: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Select the F1-optimal threshold using out-of-fold train predictions only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_proba = cross_val_predict(
        clone(clf.best_estimator_),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.arange(0.20, 0.71, 0.05)
    scores = [
        f1_score(y_train, (train_proba >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(DATA_PATH)
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split, stratified to keep the class balance consistent.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    # Grid search tuning.
    clf = tune_model(X_train, y_train)
    decision_threshold = select_threshold(clf, X_train, y_train)

    # Final predictions.
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= decision_threshold).astype(int)

    # Metrics.
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "decision_threshold": decision_threshold,
        "best_params": clf.best_params_,
        "model": type(clf.best_estimator_).__name__,
    }

    # Confusion matrix summary and save JSON.
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Save metric JSON.
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save confusion matrix plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Matrice de confusion - Churn")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Référence")
    tick_labels = ["No", "Yes"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    fig.savefig(CM_PATH)
    plt.close(fig)

    # Printable classification report.
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
    print("\nModel metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
