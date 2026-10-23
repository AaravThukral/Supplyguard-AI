"""Streamlit interface for the SupplyGuard AI demonstration."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.supplyguard_ml import CLASS_ORDER, REQUIRED_COLUMNS, explain_prediction, feature_importance, load_bundle, load_dataset, recommendations

st.set_page_config(page_title="SupplyGuard AI", page_icon="🛡️", layout="wide")

@st.cache_data
def dataset() -> pd.DataFrame:
    return load_dataset()

@st.cache_resource
def model_bundle() -> dict:
    return load_bundle()


def risk_color(label: str) -> str:
    return {"Low": "#198754", "Medium": "#d97706", "High": "#dc3545"}.get(label, "#334155")


def render_recommendations(label: str, row: pd.Series | dict) -> None:
    st.subheader("Procurement guidance")
    for item in recommendations(label, row):
        st.write(f"- {item}")
    st.caption("Guidance is a transparent decision-support rule layer, not a guarantee of an outcome.")


def numeric_input(label: str, default: int | float, minimum: int | float, maximum: int | float, step: int | float = 1) -> int | float:
    """Render a number input with one consistent numeric type for all arguments."""
    integer_input = all(isinstance(value, int) and not isinstance(value, bool) for value in (default, minimum, maximum, step))
    numeric_type = int if integer_input else float
    return st.number_input(
        label,
        min_value=numeric_type(minimum),
        max_value=numeric_type(maximum),
        value=numeric_type(default),
        step=numeric_type(step),
    )


def build_supplier_form(defaults: pd.Series | None = None) -> dict:
    get = lambda name, fallback: defaults.get(name, fallback) if defaults is not None else fallback
    left, middle, right = st.columns(3)
    with left:
        years = numeric_input("Years in business", int(get("Years_In_Business", 10)), 1, 100)
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"], index=["North", "South", "East", "West", "Central"].index(get("Region", "North")))
        capacity = numeric_input("Manufacturing capacity", int(get("Manufacturing_Capacity", 50000)), 0, 1_000_000, 1000)
        on_time = numeric_input("On-time delivery (%)", float(get("On_Time_Delivery_Percentage", 90)), 0, 100, 0.1)
        delay = numeric_input("Average delay (days)", float(get("Average_Delay_Days", 4)), 0, 365, 0.1)
        lead = numeric_input("Lead time (days)", int(get("Lead_Time_Days", 14)), 0, 365)
    with middle:
        defect = numeric_input("Defect rate (%)", float(get("Defect_Rate_Percentage", 3)), 0, 100, 0.1)
        rating = numeric_input("Customer rating", float(get("Customer_Rating", 4)), 1, 5, 0.1)
        complaints = numeric_input("Complaint count", int(get("Complaint_Count", 4)), 0, 10000)
        revenue = numeric_input("Annual revenue", float(get("Annual_Revenue", 250)), 0, 10_000_000, 10.0)
        credit = numeric_input("Credit score", int(get("Credit_Score", 700)), 0, 1000)
        debt = numeric_input("Debt ratio", float(get("Debt_Ratio", 0.3)), 0, 1, 0.01)
    with right:
        iso = st.selectbox("ISO certified", ["Yes", "No"], index=["Yes", "No"].index(get("ISO_Certified", "Yes")))
        environment = st.selectbox("Environmental compliance", ["Yes", "No"], index=["Yes", "No"].index(get("Environmental_Compliance", "Yes")))
        distance = numeric_input("Distance to factory (km)", float(get("Distance_To_Factory_KM", 300)), 0, 100_000, 10.0)
        transport = numeric_input("Transportation cost", float(get("Transportation_Cost", 15000)), 0, 10_000_000, 100.0)
        disruptions = numeric_input("Supply disruption history", int(get("Supply_Disruption_History", 1)), 0, 100)
    return {"Years_In_Business": years, "Region": region, "Manufacturing_Capacity": capacity, "On_Time_Delivery_Percentage": on_time, "Average_Delay_Days": delay, "Lead_Time_Days": lead, "Defect_Rate_Percentage": defect, "Customer_Rating": rating, "Complaint_Count": complaints, "Annual_Revenue": revenue, "Credit_Score": credit, "Debt_Ratio": debt, "ISO_Certified": iso, "Environmental_Compliance": environment, "Distance_To_Factory_KM": distance, "Transportation_Cost": transport, "Supply_Disruption_History": disruptions}


def main() -> None:
    st.title("SupplyGuard AI")
    st.caption("AI-based supplier risk assessment and procurement decision support")
    try:
        data = dataset()
        bundle = model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `python -m src.train_model` after installing requirements.")
        return
    pipeline = bundle["pipeline"]
    selected = bundle["selected_model"]
    pages = st.sidebar.radio("Navigate", ["Dashboard", "Individual assessment", "Batch assessment", "Model information"])
    if pages == "Dashboard":
        st.header("Project dashboard")
        st.write("SupplyGuard uses a reproducible scikit-learn pipeline to classify suppliers as Low, Medium, or High risk. The benchmark is synthetic and intended for academic demonstration.")
        a, b, c = st.columns(3)
        a.metric("Suppliers", f"{len(data):,}")
        b.metric("Selected model", selected)
        c.metric("Features used", len(REQUIRED_COLUMNS))
        st.subheader("Synthetic risk distribution")
        st.bar_chart(data["Risk_Label"].value_counts().reindex(CLASS_ORDER))
        st.subheader("Model comparison")
        summary = pd.DataFrame(bundle["results"]).T[["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]]
        st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)
    elif pages == "Individual assessment":
        st.header("Individual supplier assessment")
        sample = data.iloc[0] if st.checkbox("Pre-fill with a sample supplier") else None
        row = build_supplier_form(sample)
        if st.button("Assess supplier risk", type="primary"):
            frame = pd.DataFrame([row], columns=REQUIRED_COLUMNS)
            label = str(pipeline.predict(frame)[0])
            probabilities = pipeline.predict_proba(frame)[0] if hasattr(pipeline, "predict_proba") else None
            st.markdown(f"<h2 style='color:{risk_color(label)}'>Predicted risk: {label}</h2>", unsafe_allow_html=True)
            if probabilities is not None:
                st.subheader("Model output probabilities")
                st.dataframe(pd.DataFrame({"Risk class": pipeline.classes_, "Probability": probabilities}).style.format({"Probability": "{:.1%}"}), use_container_width=True)
            st.subheader("Principal associated factors")
            for factor in explain_prediction(pipeline, frame):
                direction = "associated with higher risk" if factor["contribution"] > 0 else "associated with lower risk"
                st.write(f"- **{factor['feature']}** ({direction}; contribution {factor['contribution']:.3f})")
            render_recommendations(label, row)
    elif pages == "Batch assessment":
        st.header("Batch assessment")
        st.write("Upload a CSV containing the 17 supplier feature columns. Supplier_ID is optional and preserved in the output.")
        uploaded = st.file_uploader("Supplier CSV", type="csv")
        if uploaded is not None:
            incoming = pd.read_csv(uploaded)
            missing = sorted(set(REQUIRED_COLUMNS) - set(incoming.columns))
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                valid = incoming[REQUIRED_COLUMNS].notna().all(axis=1)
                results = incoming.copy()
                results["Validation_Status"] = "Valid"
                results.loc[~valid, "Validation_Status"] = "Invalid: missing required values"
                if valid.any():
                    results.loc[valid, "Predicted_Risk"] = pipeline.predict(incoming.loc[valid, REQUIRED_COLUMNS])
                    if hasattr(pipeline, "predict_proba"):
                        probs = pipeline.predict_proba(incoming.loc[valid, REQUIRED_COLUMNS])
                        for index, cls in enumerate(pipeline.classes_): results.loc[valid, f"Probability_{cls}"] = probs[:, index]
                st.dataframe(results, use_container_width=True)
                st.download_button("Download assessment CSV", results.to_csv(index=False).encode("utf-8"), "supplyguard_assessments.csv", "text/csv")
    else:
        st.header("Model information")
        st.write(f"Selected model: **{selected}**")
        st.write("All preprocessing is inside the serialized scikit-learn pipeline and is fitted on the training split only. Supplier_ID and Risk_Label are excluded from predictors.")
        st.warning("The dataset is synthetic, labels are generated from a project-defined risk score, and evaluation measures performance on this constructed benchmark. Probabilities are model outputs, not guaranteed calibrated confidence values or causal evidence.")
        st.subheader("Global associated features")
        st.dataframe(pd.DataFrame(feature_importance(pipeline)), use_container_width=True)
        st.subheader("Evaluation metrics")
        st.dataframe(pd.DataFrame(bundle["results"]).T, use_container_width=True)

if __name__ == "__main__":
    main()
