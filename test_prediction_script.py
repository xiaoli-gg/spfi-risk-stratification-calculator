from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"


def main() -> None:
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text(encoding="utf-8"))
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
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
    input_df = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)
    model = joblib.load(MODEL_DIR / "final_svc_model.joblib")
    probability = float(model.predict_proba(input_df)[0, 1])
    threshold = float(metadata["threshold"])
    group = "Higher predicted SPFI risk" if probability >= threshold else "Lower predicted SPFI risk"
    print(f"Predicted probability: {probability:.6f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Risk group: {group}")
    if not 0.0 <= probability <= 1.0:
        raise SystemExit("Probability outside expected range.")


if __name__ == "__main__":
    main()
