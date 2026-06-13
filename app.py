from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "final_svc_model.joblib"
FEATURE_PATH = MODEL_DIR / "feature_names.json"
LABEL_PATH = MODEL_DIR / "clean_labels.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


INPUT_SPECS = {
    "Age": {"default": 65, "min": 18, "max": 100, "step": 1, "unit": "years"},
    "BMI": {"default": 23.0, "min": 10.0, "max": 50.0, "step": 0.1, "unit": "kg/m²"},
    "WBC_On_Admission": {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1, "unit": "×10⁹/L"},
    "Neutrophil_Percent": {"default": 75.0, "min": 0.0, "max": 100.0, "step": 0.1, "unit": "%"},
    "PaO2": {"default": 80.0, "min": 20.0, "max": 300.0, "step": 1.0, "unit": "mmHg"},
    "PaCO2": {"default": 40.0, "min": 10.0, "max": 150.0, "step": 1.0, "unit": "mmHg"},
    "Blood_Glucose": {"default": 7.0, "min": 1.0, "max": 50.0, "step": 0.1, "unit": "mmol/L"},
    "Oxygenation_Index": {"default": 250.0, "min": 20.0, "max": 700.0, "step": 1.0, "unit": "mmHg"},
    "CRP": {"default": 50.0, "min": 0.0, "max": 500.0, "step": 1.0, "unit": "mg/L"},
}

BINARY_FEATURES = {
    "Type2_Diabetes",
    "Glucocorticoids_Use",
    "Multi_Antibiotics_3plus",
    "Prolonged_Antibiotic_Use",
    "Surgery_Status",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_input_dataframe(feature_names: list[str], labels: dict[str, str]) -> pd.DataFrame:
    values: dict[str, float] = {}

    st.sidebar.header("Patient predictors")
    st.sidebar.caption("Use values available before the predefined prediction timepoint.")

    for feature in feature_names:
        label = labels[feature]
        if feature in BINARY_FEATURES:
            choice = st.sidebar.radio(label, ["No", "Yes"], horizontal=True, key=feature)
            values[feature] = 1 if choice == "Yes" else 0
        else:
            spec = INPUT_SPECS[feature]
            values[feature] = st.sidebar.number_input(
                f"{label} ({spec['unit']})",
                min_value=spec["min"],
                max_value=spec["max"],
                value=spec["default"],
                step=spec["step"],
                key=feature,
            )

    return pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)


def probability_chart(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[probability * 100],
            y=["Predicted probability"],
            orientation="h",
            marker_color="#D1495B" if probability >= threshold else "#2F6FB0",
            hovertemplate="%{x:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold * 100,
        line_dash="dash",
        line_color="#666666",
        annotation_text=f"Threshold {threshold * 100:.1f}%",
        annotation_position="top",
    )
    fig.update_xaxes(range=[0, 100], title="Predicted probability (%)")
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=180, margin=dict(l=10, r=20, t=35, b=35), showlegend=False)
    return fig


def approximate_linear_explanation(model, input_df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame | None:
    """Approximate feature direction for the linear SVC after preprocessing.

    This is not SHAP. It multiplies the transformed input by the linear SVC
    coefficient vector to provide a lightweight directional display.
    """
    try:
        transformed = model[:-1].transform(input_df)
        clf = model.named_steps["clf"]
        coefficients = np.ravel(clf.coef_)
        contributions = np.ravel(transformed) * coefficients
    except Exception:
        return None

    explanation = pd.DataFrame(
        {
            "Predictor": [labels[name] for name in input_df.columns],
            "Approximate contribution": contributions,
        }
    )
    explanation["Direction"] = np.where(explanation["Approximate contribution"] >= 0, "Higher risk", "Lower risk")
    explanation["Magnitude"] = explanation["Approximate contribution"].abs()
    return explanation.sort_values("Magnitude", ascending=False)


def render_model_information(metadata: dict) -> None:
    st.markdown("---")
    st.subheader("Model information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Final model", metadata["model_name"])
    col2.metric("Predictors", metadata["number_of_predictors"])
    col3.metric("Threshold", f"{metadata['threshold']:.3f}")
    st.write(f"Feature selection: {metadata['feature_selection']}")
    st.write(f"Development cohort: n = {metadata['development_cohort_n']}")
    st.write(f"Temporal validation cohort: n = {metadata['temporal_validation_n']}")
    st.info(metadata["intended_use"])


def main() -> None:
    st.set_page_config(
        page_title="SPFI Risk Stratification Calculator",
        layout="wide",
    )

    model = load_model()
    feature_names = load_json(FEATURE_PATH)
    labels = load_json(LABEL_PATH)
    metadata = load_json(METADATA_PATH)
    threshold = float(metadata["threshold"])

    st.title("SPFI Risk Stratification Calculator")
    st.subheader("Secondary Pulmonary Fungal Infection in Severe Pneumonia")
    st.caption("An educational and research-use calculator based on the final temporally validated SVC model.")

    st.warning(
        "This tool is intended for research and educational demonstration only. "
        "It is not a standalone diagnostic tool and should not be used to guide antifungal "
        "treatment decisions without clinical judgement and further validation."
    )

    input_df = build_input_dataframe(feature_names, labels)

    left, right = st.columns([1.1, 1.0])
    with left:
        st.subheader("Input summary")
        display_df = input_df.rename(columns=labels)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Prediction")
        predict = st.button("Predict SPFI risk", type="primary", use_container_width=True)
        if predict:
            probability = float(model.predict_proba(input_df)[0, 1])
            risk_group = "Higher predicted SPFI risk" if probability >= threshold else "Lower predicted SPFI risk"
            st.metric("Predicted probability of SPFI", f"{probability * 100:.1f}%")
            if probability >= threshold:
                st.error(risk_group)
            else:
                st.success(risk_group)
            st.write(f"Threshold: {threshold:.3f}")
            st.caption(
                "The threshold was determined using the Youden index based on "
                "training-set cross-validated predicted probabilities."
            )
            st.plotly_chart(probability_chart(probability, threshold), use_container_width=True)

            st.markdown("**Temporal validation performance**")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("AUROC", f"{metadata['temporal_validation_AUROC']:.3f}")
            m2.metric("AUPRC", f"{metadata['temporal_validation_AUPRC']:.3f}")
            m3.metric("Sensitivity", f"{metadata['temporal_validation_sensitivity']:.3f}")
            m4.metric("Specificity", f"{metadata['temporal_validation_specificity']:.3f}")
            m5.metric("Brier score", f"{metadata['temporal_validation_Brier']:.3f}")

            with st.expander("Individual explanation"):
                show_explanation = st.checkbox("Show approximate linear model-direction display")
                if show_explanation:
                    explanation = approximate_linear_explanation(model, input_df, labels)
                    if explanation is None:
                        st.info("Individual SHAP explanation is not enabled in this demonstration version.")
                    else:
                        st.caption(
                            "This lightweight display uses the fitted linear SVC coefficients after preprocessing. "
                            "It is a directional approximation, not a SHAP explanation and not causal evidence."
                        )
                        top_positive = explanation[explanation["Approximate contribution"] > 0].head(5)
                        top_negative = explanation[explanation["Approximate contribution"] < 0].head(5)
                        e1, e2 = st.columns(2)
                        with e1:
                            st.markdown("**Top higher-risk directions**")
                            st.dataframe(
                                top_positive[["Predictor", "Approximate contribution"]],
                                hide_index=True,
                                use_container_width=True,
                            )
                        with e2:
                            st.markdown("**Top lower-risk directions**")
                            st.dataframe(
                                top_negative[["Predictor", "Approximate contribution"]],
                                hide_index=True,
                                use_container_width=True,
                            )
                        plot_df = pd.concat([top_positive, top_negative]).sort_values("Approximate contribution")
                        if not plot_df.empty:
                            fig = go.Figure(
                                go.Bar(
                                    x=plot_df["Approximate contribution"],
                                    y=plot_df["Predictor"],
                                    orientation="h",
                                    marker_color=np.where(plot_df["Approximate contribution"] >= 0, "#D1495B", "#2F6FB0"),
                                )
                            )
                            fig.update_layout(height=360, margin=dict(l=10, r=20, t=10, b=30))
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Individual SHAP explanation is not enabled in this demonstration version.")

    render_model_information(metadata)

    st.markdown("---")
    st.caption(
        "Disclaimer: This calculator is not intended to diagnose SPFI, replace microbiological or "
        "radiological evaluation, or guide antifungal therapy. Clinical use requires prospective "
        "multicenter validation and appropriate regulatory approval."
    )


if __name__ == "__main__":
    main()
