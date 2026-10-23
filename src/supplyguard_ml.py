"""Reusable SupplyGuard AI training, inference, and explanation utilities."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "Risk_Label"
ID_COLUMN = "Supplier_ID"
CLASS_ORDER = ["Low", "Medium", "High"]
CATEGORICAL_COLUMNS = ["Region", "ISO_Certified", "Environmental_Compliance"]
REQUIRED_COLUMNS = [
    "Years_In_Business", "Region", "Manufacturing_Capacity", "On_Time_Delivery_Percentage",
    "Average_Delay_Days", "Lead_Time_Days", "Defect_Rate_Percentage", "Customer_Rating",
    "Complaint_Count", "Annual_Revenue", "Credit_Score", "Debt_Ratio", "ISO_Certified",
    "Environmental_Compliance", "Distance_To_Factory_KM", "Transportation_Cost",
    "Supply_Disruption_History",
]
NUMERIC_COLUMNS = [column for column in REQUIRED_COLUMNS if column not in CATEGORICAL_COLUMNS]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2 compatibility
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("numeric", StandardScaler(), NUMERIC_COLUMNS), ("categorical", encoder, CATEGORICAL_COLUMNS)],
        remainder="drop",
    )


def model_catalog() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    }


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path else project_root() / "data" / "suppliers.csv"
    data = pd.read_csv(dataset_path)
    missing = set(REQUIRED_COLUMNS + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return data


def split_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data[REQUIRED_COLUMNS].copy(), data[TARGET].astype(str)


def build_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, elapsed: float) -> dict[str, Any]:
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=CLASS_ORDER).tolist(),
        "class_order": CLASS_ORDER,
        "training_seconds": round(elapsed, 4),
    }


def train_all(data: pd.DataFrame | None = None) -> tuple[dict[str, Pipeline], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = data if data is not None else load_dataset()
    X, y = split_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    fitted, results = {}, {}
    for name, estimator in model_catalog().items():
        pipeline = build_pipeline(estimator)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_pipeline(pipeline, X_test, y_test, time.perf_counter() - started)
        fitted[name] = pipeline
    return fitted, results, X_train, X_test, y_train, y_test


def choose_model(results: dict[str, Any]) -> str:
    return max(results, key=lambda name: (results[name]["macro_f1"], results[name]["accuracy"], name == "Logistic Regression"))


def _clean_feature_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


def feature_importance(pipeline: Pipeline, top_n: int = 10) -> list[dict[str, Any]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.mean(np.abs(np.asarray(model.coef_)), axis=0)
    else:
        return []
    order = np.argsort(values)[::-1][:top_n]
    return [{"feature": _clean_feature_name(str(names[i])), "importance": float(values[i])} for i in order]


def explain_prediction(pipeline: Pipeline, row: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    transformed = preprocessor.transform(row[REQUIRED_COLUMNS])
    predicted = str(pipeline.predict(row[REQUIRED_COLUMNS])[0])
    if hasattr(model, "coef_"):
        class_index = list(model.classes_).index(predicted)
        contributions = np.asarray(transformed[0]).ravel() * np.asarray(model.coef_[class_index]).ravel()
    elif hasattr(model, "feature_importances_"):
        contributions = np.asarray(transformed[0]).ravel() * np.asarray(model.feature_importances_).ravel()
    else:
        return []
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [{"feature": _clean_feature_name(str(names[i])), "contribution": float(contributions[i])} for i in order]


def recommendations(label: str, row: pd.Series | dict[str, Any]) -> list[str]:
    values = row if isinstance(row, dict) else row.to_dict()
    advice = {
        "High": ["Perform a detailed supplier review before relying on this supplier.", "Examine delivery reliability, quality defects, financial indicators, and disruption history.", "Add closer monitoring and consider contingency or alternate-supplier planning."],
        "Medium": ["Use targeted monitoring and identify the weaker performance areas.", "Agree corrective actions with the supplier and schedule a follow-up review."],
        "Low": ["Continue routine supplier monitoring and standard performance reviews."],
    }.get(label, ["Review the input and model output before making a procurement decision."])
    triggers = []
    if float(values.get("On_Time_Delivery_Percentage", 100)) < 85: triggers.append("on-time delivery is below 85%")
    if float(values.get("Defect_Rate_Percentage", 0)) > 8: triggers.append("defect rate is above 8%")
    if float(values.get("Credit_Score", 850)) < 600: triggers.append("credit score is below 600")
    if int(values.get("Supply_Disruption_History", 0)) >= 4: triggers.append("disruption history is elevated")
    if triggers: advice.append("Data-linked factors to review: " + ", ".join(triggers) + ".")
    return advice


def save_bundle(path: str | Path, pipeline: Pipeline, selected_model: str, results: dict[str, Any]) -> None:
    bundle = {"pipeline": pipeline, "selected_model": selected_model, "results": results, "required_columns": REQUIRED_COLUMNS, "class_order": CLASS_ORDER, "random_state": RANDOM_STATE}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path | None = None) -> dict[str, Any]:
    artifact = Path(path) if path else project_root() / "models" / "supplyguard_model.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact}. Run `python -m src.train_model` first.")
    return joblib.load(artifact)
