from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


APP_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = APP_DIR / "model"


def load_feature_names() -> list[str]:
    return json.loads((MODEL_DIR / "feature_names.json").read_text(encoding="utf-8"))


def example_input(feature_names: list[str]) -> pd.DataFrame:
    values = {
        "Age": 65,
        "BMI": 23.0,
        "Type2_Diabetes": 0,
        "Glucocorticoids_Use": 0,
        "Multi_Antibiotics_3plus": 0,
        "Prolonged_Antibiotic_Use": 0,
        "Surgery_Status": 0,
        "WBC_On_Admission": 8.0,
        "Neutrophil_Percent": 75.0,
        "PaO2": 80.0,
        "PaCO2": 40.0,
        "Blood_Glucose": 7.0,
        "Oxygenation_Index": 250.0,
        "CRP": 50.0,
    }
    return pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)


def test_feature_names_have_final_14_predictors() -> None:
    feature_names = load_feature_names()
    assert len(feature_names) == 14
    assert feature_names == [
        "Age",
        "BMI",
        "Type2_Diabetes",
        "Glucocorticoids_Use",
        "Multi_Antibiotics_3plus",
        "Prolonged_Antibiotic_Use",
        "Surgery_Status",
        "WBC_On_Admission",
        "Neutrophil_Percent",
        "PaO2",
        "PaCO2",
        "Blood_Glucose",
        "Oxygenation_Index",
        "CRP",
    ]


def test_model_file_exists() -> None:
    assert (MODEL_DIR / "final_svc_model.joblib").exists()


def test_example_input_predicts_probability_between_zero_and_one() -> None:
    feature_names = load_feature_names()
    model = joblib.load(MODEL_DIR / "final_svc_model.joblib")
    input_df = example_input(feature_names)
    probability = float(model.predict_proba(input_df)[0, 1])
    assert 0.0 <= probability <= 1.0


def test_threshold_is_final_value() -> None:
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["threshold"] == 0.293
