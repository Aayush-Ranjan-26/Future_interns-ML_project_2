# Telco Customer Churn – ML Pipeline

End-to-end, reproducible pipeline to explore, clean, model, and evaluate Telco customer churn using a locally generated dataset (7043 rows, 21 columns) with realistic distributions. The pipeline trains multiple models, handles class imbalance with SMOTE, and saves metrics, plots, and the best model artifact.

## Quickstart

1) Install dependencies (already run in this workspace):
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn openpyxl jupyter pytest scipy plotly kaleido --upgrade
```

2) Run the pipeline from the project root:
```
python telco_churn_pipeline.py
```

3) Review outputs in `outputs/`:
- `metrics.json` – accuracy/precision/recall/f1/roc_auc for the best model
- `classification_report.txt`
- `confusion_matrix.png`, `roc_curve.png`
- `feature_importance.csv` (if available)
- `model.joblib` – full preprocessing + model pipeline

## Data

- Source file: `data/telco_raw.csv` (synthetic but schema-aligned to the IBM Telco churn dataset: 7043 rows, 21 columns).
- Cleaning steps: strip strings, coerce `TotalCharges` to numeric, median-impute numeric columns, normalize target casing.

## Modeling

- Preprocessing: One-hot encode categoricals; standardize numerics.
- Imbalance handling: `SMOTE`.
- Models tried: Logistic Regression (`class_weight="balanced"`), Random Forest, XGBoost (`scale_pos_weight`).
- Model selection: chooses the highest f1 (tie-broken by ROC AUC) on a stratified train/test split.
