# SPFI Risk Stratification Calculator

## Intended use

This Streamlit application provides a single-patient risk stratification demonstration for secondary pulmonary fungal infection in patients with severe pneumonia. It is intended for research discussion, departmental teaching, and educational demonstration only.

It is not a standalone diagnostic tool and must not be used to guide antifungal treatment decisions without clinical judgement, microbiological evaluation, radiological evaluation, and further prospective validation.

## Model summary

- Final model: Support Vector Classifier
- Feature selection: RFECV with one-standard-error rule
- Number of predictors: 14
- Classification threshold: 0.293
- Development cohort: n = 659
- Training set: n = 463
- Internal hold-out test set: n = 196
- Temporal validation cohort: n = 150
- Temporal validation AUROC: 0.766
- Temporal validation AUPRC: 0.659
- Temporal validation sensitivity: 0.780
- Temporal validation specificity: 0.600
- Temporal validation Brier score: 0.186

The app loads the frozen final SVC pipeline from `model/final_svc_model.joblib` and performs one-row prediction with `predict_proba`.

## Predictor list

The app accepts only the final 14 predictors:

1. Age
2. Body mass index
3. Type 2 diabetes
4. Glucocorticoid use
5. Three or more antibiotics
6. Prolonged antibiotic use
7. Surgery before prediction
8. White blood cell count on admission
9. Neutrophil percentage
10. PaO2
11. PaCO2
12. Blood glucose
13. Oxygenation index
14. C-reactive protein

The internal feature names and order are stored in `model/feature_names.json`.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## How to deploy to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload the contents of this `streamlit_spfi_predictor/` directory to the repository root.
3. Confirm that the repository includes `app.py`, `requirements.txt`, `README.md`, and the `model/` directory.
4. In Streamlit Community Cloud, create a new app from the repository.
5. Set the main file path to `app.py`.
6. Deploy.

The app uses relative paths and does not require raw training data.

## Files included

- `app.py`: Streamlit application.
- `requirements.txt`: Python dependencies.
- `model/final_svc_model.joblib`: frozen final SVC pipeline.
- `model/feature_names.json`: feature names in model input order.
- `model/clean_labels.json`: display labels.
- `model/model_metadata.json`: model metadata and temporal-validation performance.
- `tests/test_prediction_input.py`: pytest checks.
- `test_prediction_script.py`: command-line smoke test.
- `assets/`: placeholder assets.

## Important disclaimer

This calculator is not intended to diagnose SPFI, replace microbiological or radiological evaluation, or guide antifungal therapy. Clinical use requires prospective multicenter validation and appropriate regulatory approval.

## Citation placeholder

Please cite the associated manuscript after publication.
