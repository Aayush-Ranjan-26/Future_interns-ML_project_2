from pathlib import Path

import pandas as pd


def test_telco_schema():
    path = Path("data/telco_raw.csv")
    assert path.exists(), "telco_raw.csv is missing"

    df = pd.read_csv(path)
    assert df.shape == (7043, 21)

    expected_cols = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
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
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    assert list(df.columns) == expected_cols

