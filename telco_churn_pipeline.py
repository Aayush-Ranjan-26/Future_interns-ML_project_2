import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - xgboost is optional at runtime
    XGBClassifier = None  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DATA_PATH = Path("data/telco_raw.csv")
OUTPUT_DIR = Path("outputs")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset at {path}, but it was not found.")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # Strip whitespace from string columns
    obj_cols = cleaned.select_dtypes(include="object").columns
    cleaned[obj_cols] = cleaned[obj_cols].apply(lambda col: col.str.strip())

    # Coerce numeric column and fill any missing values
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    num_cols = cleaned.select_dtypes(exclude="object").columns
    cleaned[num_cols] = cleaned[num_cols].fillna(cleaned[num_cols].median())

    # The target should be a clean categorical label
    cleaned["Churn"] = cleaned["Churn"].str.title()
    return cleaned


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    target = "Churn"
    feature_df = df.drop(columns=[target])
    cat_cols = feature_df.select_dtypes(include="object").columns.tolist()
    num_cols = feature_df.select_dtypes(exclude="object").columns.tolist()

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, cat_cols),
            ("numeric", numeric, num_cols),
        ]
    )

    return preprocessor, cat_cols, num_cols


def get_model_display_name(name: str) -> str:
    """Map internal model names to display names."""
    mapping = {
        "log_reg": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost",
    }
    return mapping.get(name, name.title())


def get_models(pos_weight: float) -> Dict[str, object]:
    models: Dict[str, object] = {
        "log_reg": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "rf": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }
    if XGBClassifier is not None:
        models["xgb"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=pos_weight,
        )
    return models


def evaluate_model(
    name: str,
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float]:
    # Cross-validation on training set
    cv_scores = cross_val_score(
        model_pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )
    cv_accuracy = cv_scores.mean()
    
    # Train and evaluate on test set
    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_test)
    proba = model_pipeline.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )

    metrics = {
        "model": name,
        "cv_accuracy": cv_accuracy,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, proba),
    }

    return metrics


def plot_confusion(y_true: np.ndarray, preds: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_roc(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model_pipeline, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def extract_feature_importance(
    fitted_pipeline: Pipeline,
    cat_cols: List[str],
    num_cols: List[str],
) -> pd.DataFrame:
    preprocessor: ColumnTransformer = fitted_pipeline.named_steps["preprocess"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    feature_names = np.concatenate([cat_feature_names, num_cols])

    estimator = fitted_pipeline.named_steps["model"]
    importances = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).ravel()

    if importances is None:
        return pd.DataFrame()

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return fi


def save_artifacts(
    metrics: Dict[str, float],
    report: str,
    importance: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    model_pipeline: Pipeline,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "classification_report.txt").write_text(report)

    if not importance.empty:
        importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plot_confusion(y_test, preds, OUTPUT_DIR / "confusion_matrix.png")
    plot_roc(model_pipeline, X_test, y_test, OUTPUT_DIR / "roc_curve.png")

    # Save the full pipeline for reuse
    try:
        import joblib

        joblib.dump(model_pipeline, OUTPUT_DIR / "model.joblib")
    except Exception as exc:  # pragma: no cover - joblib errors should not crash pipeline
        logging.warning("Skipping model persistence: %s", exc)


def main() -> None:
    df = load_data()
    df = clean_data(df)

    target_col = "Churn"
    y = df[target_col].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[target_col])

    preprocessor, cat_cols, num_cols = build_preprocessor(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    models = get_models(pos_weight)
    results: List[Dict[str, float]] = []
    fitted_models: Dict[str, Pipeline] = {}

    # Train and evaluate all models with cross-validation
    print("\n" + "=" * 50)
    for name, estimator in models.items():
        display_name = get_model_display_name(name)
        print(f"Training {display_name} with default parameters")
        
        pipe = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("smote", SMOTE(random_state=42)),
                ("model", estimator),
            ]
        )
        metrics = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results.append(metrics)
        fitted_models[name] = pipe
        
        print(f"{display_name} cross-validation accuracy: {metrics['cv_accuracy']:.2f}")
        print("-" * 50)

    # Select best model
    best = max(results, key=lambda m: (m["f1"], m["roc_auc"]))
    best_name = best["model"]
    best_model = fitted_models[best_name]

    # Fit best model fully and evaluate
    best_model.fit(X_train, y_train)
    best_preds = best_model.predict(X_test)
    best_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate metrics for best model
    accuracy = accuracy_score(y_test, best_preds)
    cm = confusion_matrix(y_test, best_preds)
    report = classification_report(
        y_test, best_preds, target_names=["No", "Yes"], zero_division=0
    )

    # Print final results in desired format
    print("\n" + "=" * 50)
    print("FINAL MODEL EVALUATION")
    print("=" * 50)
    print(f"\nAccuracy Score:")
    print(f"{accuracy}")
    
    print(f"\nConfusion Matrix:")
    print(f"[[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f" [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    
    print(f"\nClassification Report:")
    print(report)
    print("=" * 50 + "\n")

    importance = extract_feature_importance(best_model, cat_cols, num_cols)

    save_artifacts(
        metrics=best,
        report=report,
        importance=importance,
        X_test=X_test,
        y_test=y_test.to_numpy(),
        preds=best_preds,
        model_pipeline=best_model,
    )


if __name__ == "__main__":
    main()

