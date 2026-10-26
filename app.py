"""SupplyGuard AI enterprise procurement analytics interface."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.reporting import alert_records, build_batch_pdf, build_supplier_pdf
from src.supplyguard_ml import (
    CLASS_ORDER,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    explain_prediction,
    feature_importance,
    load_bundle,
    load_dataset,
    recommendations,
)

st.set_page_config(page_title="SupplyGuard AI", page_icon="SG", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root { --navy:#07111f; --panel:#0f1d2e; --panel2:#14263b; --emerald:#10b981; --amber:#f59e0b; --coral:#fb7185; --muted:#94a3b8; }
    .stApp { background: linear-gradient(135deg, #07111f 0%, #0b1727 55%, #0d1e31 100%); color: #e2e8f0; }
    [data-testid="stSidebar"] { background: #081321; border-right: 1px solid #1e334b; }
    [data-testid="stSidebar"] * { color: #dbeafe; }
    h1, h2, h3 { color: #f8fafc !important; letter-spacing: -0.02em; }
    .eyebrow { color:#34d399; font-size:.74rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.35rem; }
    .hero { background:linear-gradient(120deg,#10253a,#0d1a2d); border:1px solid #21415d; border-radius:18px; padding:1.5rem 1.7rem; margin-bottom:1rem; }
    .hero p { color:#a9bed2; max-width:850px; margin-bottom:0; }
    .kpi { background:rgba(15,29,46,.92); border:1px solid #1f3b55; border-radius:14px; padding:1rem 1.1rem; min-height:112px; }
    .kpi-label { color:#91a7bc; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
    .kpi-value { color:#f8fafc; font-size:1.8rem; font-weight:750; margin-top:.35rem; }
    .kpi-note { color:#647f98; font-size:.75rem; margin-top:.2rem; }
    .risk-high { color:#fb7185 !important; font-weight:750; }
    .risk-medium { color:#fbbf24 !important; font-weight:750; }
    .risk-low { color:#34d399 !important; font-weight:750; }
    .section-note { color:#9fb2c5; font-size:.9rem; }
    div[data-testid="stDataFrame"] { border:1px solid #1f3b55; border-radius:12px; overflow:hidden; }
    .alert-card { background:#162334; border-left:4px solid #f59e0b; border-radius:8px; padding:.7rem .9rem; margin:.35rem 0; }
    .alert-card.high { border-left-color:#fb7185; }
    .alert-card.medium { border-left-color:#f59e0b; }
    .muted-box { background:#0d1a2a; border:1px solid #1f3b55; border-radius:12px; padding:1rem; color:#a9bed2; }
    .stButton > button[kind="primary"] { background:#059669; border-color:#10b981; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource
def model_bundle() -> dict[str, Any]:
    return load_bundle()


def numeric_input(label: str, default: int | float, minimum: int | float, maximum: int | float, step: int | float = 1) -> int | float:
    """Render a number input with one consistent type for all Streamlit numeric arguments."""
    integer_input = all(isinstance(value, int) and not isinstance(value, bool) for value in (default, minimum, maximum, step))
    numeric_type = int if integer_input else float
    return st.number_input(label, min_value=numeric_type(minimum), max_value=numeric_type(maximum), value=numeric_type(default), step=numeric_type(step))


def risk_class(value: str) -> str:
    return str(value).strip().title()


def risk_css(value: str) -> str:
    return {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(risk_class(value), "")


def hero(title: str, subtitle: str, eyebrow: str = "SupplyGuard AI") -> None:
    st.markdown(f"<div class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>", unsafe_allow_html=True)


def predict_dataset(data: pd.DataFrame, pipeline: Any) -> pd.DataFrame:
    result = data.copy()
    features = result[REQUIRED_COLUMNS]
    result["Predicted_Risk"] = pipeline.predict(features)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        for index, label in enumerate(pipeline.classes_):
            result[f"Probability_{label}"] = probabilities[:, index]
    return result


def probability_map(pipeline: Any, frame: pd.DataFrame) -> dict[str, float]:
    if not hasattr(pipeline, "predict_proba"):
        return {}
    values = pipeline.predict_proba(frame[REQUIRED_COLUMNS])[0]
    return {str(label): float(value) for label, value in zip(pipeline.classes_, values)}


def build_supplier_form(defaults: pd.Series | None = None) -> dict[str, Any]:
    def get(name: str, fallback: Any) -> Any:
        return defaults.get(name, fallback) if defaults is not None else fallback

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


def render_risk_badge(label: str) -> None:
    st.markdown(f"<div class='{risk_css(label)}' style='font-size:1.6rem'>Predicted risk: {risk_class(label)}</div>", unsafe_allow_html=True)


def render_alerts(row: pd.Series | dict[str, Any]) -> None:
    alerts = alert_records(row)
    if not alerts:
        st.success("No transparent rule-based alert thresholds triggered for this supplier.")
        return
    for alert in alerts:
        st.markdown(f"<div class='alert-card {alert['severity'].lower()}'><b>{alert['category']}</b><br/>{alert['message']}</div>", unsafe_allow_html=True)
    st.caption("These are documented threshold alerts, separate from the machine-learning prediction.")


def render_guidance(label: str, row: pd.Series | dict[str, Any]) -> list[str]:
    guidance = recommendations(risk_class(label), row)
    st.subheader("Procurement guidance")
    for item in guidance:
        st.write(f"- {item}")
    st.caption("Guidance is a transparent decision-support rule layer, not a guarantee of an outcome.")
    return guidance


def render_model_output(pipeline: Any, frame: pd.DataFrame, selected_model: str, include_pdf: bool = False) -> None:
    label = risk_class(pipeline.predict(frame[REQUIRED_COLUMNS])[0])
    probabilities = probability_map(pipeline, frame)
    factors = explain_prediction(pipeline, frame)
    st.divider()
    render_risk_badge(label)
    st.caption(f"Selected model: {selected_model}. Model output is based on the synthetic benchmark pipeline.")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Model output probabilities")
        if probabilities:
            st.dataframe(pd.DataFrame({"Risk class": list(probabilities), "Probability": list(probabilities.values())}).style.format({"Probability": "{:.1%}"}), use_container_width=True, hide_index=True)
        else:
            st.info("This model does not provide probabilities.")
        st.subheader("Model-associated factors")
        for factor in factors:
            direction = "associated with higher risk" if factor["contribution"] > 0 else "associated with lower risk"
            st.write(f"- **{factor['feature']}** — {direction} ({factor['contribution']:.3f})")
        st.caption("Factors are model associations, not causal explanations. Global importance must not be read as an individual explanation.")
    with right:
        render_alerts(frame.iloc[0])
        guidance = render_guidance(label, frame.iloc[0])
        if include_pdf:
            pdf = build_supplier_pdf(frame.iloc[0], label, probabilities, factors, guidance, selected_model)
            st.download_button("Download supplier PDF report", pdf, f"supplyguard_{frame.iloc[0].get('Supplier_ID', 'assessment')}.pdf", "application/pdf", type="primary")


def page_overview(enriched: pd.DataFrame, bundle: dict[str, Any], selected_model: str) -> None:
    hero("Procurement command center", "A decision-support view of supplier risk, operational exposure, and review priorities. Dataset labels and model predictions are shown separately.", "Overview")
    high = int((enriched["Predicted_Risk"] == "High").sum())
    avg_delivery = enriched["On_Time_Delivery_Percentage"].mean()
    avg_defects = enriched["Defect_Rate_Percentage"].mean()
    cols = st.columns(4)
    with cols[0]: kpi("Total suppliers", f"{len(enriched):,}", "Loaded from suppliers.csv")
    with cols[1]: kpi("Predicted high risk", f"{high:,} ({high / len(enriched):.1%})", "Current model output")
    with cols[2]: kpi("Avg. on-time delivery", f"{avg_delivery:.1f}%", "Dataset operating metric")
    with cols[3]: kpi("Avg. defect rate", f"{avg_defects:.2f}%", "Dataset quality metric")
    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Predicted risk distribution")
        st.bar_chart(enriched["Predicted_Risk"].value_counts().reindex(CLASS_ORDER, fill_value=0), color="#10b981")
        st.caption("Predicted classes from the selected saved model.")
        st.subheader("Dataset label distribution")
        st.bar_chart(enriched["Risk_Label"].value_counts().reindex(CLASS_ORDER, fill_value=0), color="#64748b")
        st.caption("Synthetic labels generated by the project-defined risk score.")
    with right:
        st.subheader("Suppliers deserving attention")
        attention = enriched[enriched["Predicted_Risk"] == "High"].copy()
        if attention.empty:
            attention = enriched.sort_values(["Defect_Rate_Percentage", "On_Time_Delivery_Percentage"], ascending=[False, True]).head(10)
        attention = attention.sort_values(["Defect_Rate_Percentage", "On_Time_Delivery_Percentage"], ascending=[False, True]).head(10)
        st.dataframe(attention[["Supplier_ID", "Predicted_Risk", "Risk_Label", "Region", "On_Time_Delivery_Percentage", "Defect_Rate_Percentage", "Credit_Score", "Supply_Disruption_History"]], use_container_width=True, hide_index=True)
        st.caption("Risk_Label is the synthetic dataset label; Predicted_Risk is the model output.")
    st.subheader("Model snapshot")
    metrics = pd.DataFrame(bundle["results"]).T
    st.dataframe(metrics[["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]].style.format("{:.3f}"), use_container_width=True)


def page_suppliers(enriched: pd.DataFrame) -> None:
    hero("Supplier directory", "Search, filter, and inspect the supplier population without treating a risk class as a universal procurement winner.", "Suppliers")
    search = st.text_input("Search supplier ID", placeholder="e.g. SUP0042")
    c1, c2, c3, c4 = st.columns(4)
    with c1: risk = st.multiselect("Predicted risk", CLASS_ORDER, default=[])
    with c2: region = st.multiselect("Region", sorted(enriched["Region"].dropna().unique()), default=[])
    with c3: min_delivery = st.slider("Minimum on-time delivery", 0, 100, 0)
    with c4: max_defect = st.slider("Maximum defect rate", 0.0, 20.0, 20.0, 0.1)
    filtered = enriched.copy()
    if search: filtered = filtered[filtered["Supplier_ID"].astype(str).str.contains(search, case=False, na=False)]
    if risk: filtered = filtered[filtered["Predicted_Risk"].isin(risk)]
    if region: filtered = filtered[filtered["Region"].isin(region)]
    filtered = filtered[(filtered["On_Time_Delivery_Percentage"] >= min_delivery) & (filtered["Defect_Rate_Percentage"] <= max_defect)]
    st.write(f"Showing **{len(filtered):,}** of {len(enriched):,} suppliers")
    columns = ["Supplier_ID", "Predicted_Risk", "Risk_Label", "Region", "On_Time_Delivery_Percentage", "Average_Delay_Days", "Lead_Time_Days", "Defect_Rate_Percentage", "Customer_Rating", "Credit_Score", "Supply_Disruption_History"]
    st.dataframe(filtered[columns].sort_values(["Predicted_Risk", "Defect_Rate_Percentage"], ascending=[True, False]), use_container_width=True, hide_index=True)
    st.download_button("Export filtered supplier directory", filtered.to_csv(index=False).encode("utf-8"), "supplyguard_supplier_directory.csv", "text/csv")


def page_assessments(data: pd.DataFrame, pipeline: Any, selected_model: str) -> None:
    hero("Supplier assessments", "Assess an existing supplier or enter a prospective supplier profile. The prediction, explanations, alerts, and guidance remain separate layers.", "Assessments")
    assessment_type = st.radio("Assessment type", ["Individual supplier", "Batch CSV"], horizontal=True)
    if assessment_type == "Batch CSV":
        st.write("Upload a CSV containing the 17 supplier feature columns. Supplier_ID and Risk_Label are optional and preserved when present.")
        uploaded = st.file_uploader("Supplier CSV", type="csv", key="assessment_batch_upload")
        if uploaded is None:
            st.info("Upload a CSV to validate and assess multiple suppliers.")
            return
        try:
            incoming = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"The uploaded file could not be read as CSV: {exc}")
            return
        missing = sorted(set(REQUIRED_COLUMNS) - set(incoming.columns))
        if missing:
            st.error(f"Missing required columns: {missing}")
            return
        validated = incoming.copy()
        invalid_reasons = pd.Series("Valid", index=validated.index, dtype="object")
        for column in NUMERIC_COLUMNS:
            converted = pd.to_numeric(validated[column], errors="coerce")
            invalid = converted.isna() & validated[column].notna()
            invalid_reasons.loc[invalid] = f"Invalid numeric value in {column}"
            validated[column] = converted
        invalid_reasons.loc[validated[REQUIRED_COLUMNS].isna().any(axis=1) & (invalid_reasons == "Valid")] = "Missing required value"
        results = incoming.copy()
        results["Validation_Status"] = invalid_reasons
        valid = invalid_reasons == "Valid"
        if valid.any():
            valid_features = validated.loc[valid, REQUIRED_COLUMNS]
            results.loc[valid, "Predicted_Risk"] = pipeline.predict(valid_features)
            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(valid_features)
                for index, label in enumerate(pipeline.classes_):
                    results.loc[valid, f"Probability_{label}"] = probabilities[:, index]
        st.write(f"Valid rows: **{int(valid.sum())}** | Invalid rows: **{int((~valid).sum())}")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("Download batch assessment CSV", results.to_csv(index=False).encode("utf-8"), "supplyguard_batch_assessments.csv", "text/csv", type="primary")
        if valid.any():
            try:
                batch_pdf = build_batch_pdf(results.loc[valid], selected_model)
                st.download_button("Download batch PDF summary", batch_pdf, "supplyguard_batch_assessments.pdf", "application/pdf")
            except RuntimeError as exc:
                st.warning(str(exc))
        return

    mode = st.radio("Assessment source", ["Existing supplier", "Manual profile"], horizontal=True)
    defaults = None
    supplier_id = None
    if mode == "Existing supplier":
        supplier_id = st.selectbox("Select supplier", data["Supplier_ID"].tolist())
        defaults = data.loc[data["Supplier_ID"] == supplier_id].iloc[0]
    else:
        st.info("Manual profiles are assessed by the saved pipeline but are not added to the dataset.")
    with st.form("assessment_form"):
        values = build_supplier_form(defaults)
        submitted = st.form_submit_button("Run supplier assessment", type="primary")
    if submitted:
        if supplier_id: values["Supplier_ID"] = supplier_id
        frame = pd.DataFrame([{column: values[column] for column in REQUIRED_COLUMNS}], columns=REQUIRED_COLUMNS)
        if supplier_id:
            frame["Supplier_ID"] = supplier_id
        render_model_output(pipeline, frame, selected_model, include_pdf=True)


def page_comparison(enriched: pd.DataFrame, pipeline: Any, selected_model: str) -> None:
    hero("Supplier comparison", "Compare two or three suppliers across risk, operational performance, quality, and financial indicators. The tool surfaces trade-offs instead of declaring a universal winner.", "Supplier Comparison")
    selected_ids = st.multiselect("Select two or three suppliers", enriched["Supplier_ID"].tolist(), max_selections=3)
    if len(selected_ids) < 2:
        st.info("Select at least two suppliers to see a comparison.")
        return
    selected = enriched[enriched["Supplier_ID"].isin(selected_ids)].set_index("Supplier_ID").loc[selected_ids]
    metrics = ["Predicted_Risk", "On_Time_Delivery_Percentage", "Average_Delay_Days", "Lead_Time_Days", "Defect_Rate_Percentage", "Customer_Rating", "Annual_Revenue", "Credit_Score", "Debt_Ratio", "Transportation_Cost", "Supply_Disruption_History"]
    st.subheader("Trade-off matrix")
    st.dataframe(selected[metrics].T, use_container_width=True)
    st.subheader("Model output probabilities")
    probability_columns = [column for column in selected.columns if column.startswith("Probability_")]
    if probability_columns: st.dataframe(selected[probability_columns].T.style.format("{:.1%}"), use_container_width=True)
    st.subheader("Rule-based review signals")
    for supplier_id, row in selected.iterrows():
        alerts = alert_records(row)
        st.write(f"**{supplier_id}** — {len(alerts)} alert(s): " + ("; ".join(alert["message"] for alert in alerts) if alerts else "none"))
    st.caption("Risk class is one input to procurement review; cost, reliability, quality, financial indicators, and context should also be considered.")


def page_intelligence(enriched: pd.DataFrame) -> None:
    hero("Risk intelligence", "A transparent review-priority layer based on documented thresholds. Alerts are not machine-learning predictions and do not represent live monitoring.", "Risk Intelligence")
    records = []
    for _, row in enriched.iterrows():
        for alert in alert_records(row):
            records.append({"Supplier_ID": row["Supplier_ID"], "Predicted_Risk": row["Predicted_Risk"], **alert})
    alerts = pd.DataFrame(records)
    if alerts.empty:
        st.success("No alert thresholds are triggered in the current dataset.")
        return
    c1, c2 = st.columns(2)
    with c1: severity = st.multiselect("Severity", ["High", "Medium"], default=[])
    with c2: category = st.multiselect("Alert category", sorted(alerts["category"].unique()), default=[])
    filtered = alerts.copy()
    if severity: filtered = filtered[filtered["severity"].isin(severity)]
    if category: filtered = filtered[filtered["category"].isin(category)]
    st.metric("Flagged suppliers", filtered["Supplier_ID"].nunique())
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Export risk alert list", filtered.to_csv(index=False).encode("utf-8"), "supplyguard_risk_alerts.csv", "text/csv")
    st.caption("Thresholds are documented in src/reporting.py. This local application does not monitor suppliers over time.")


def page_reports(enriched: pd.DataFrame, pipeline: Any, selected_model: str) -> None:
    hero("Reports", "Generate a supplier-level PDF assessment or export the current prediction and alert data for follow-up workflows.", "Reports")
    supplier_id = st.selectbox("Supplier for PDF report", enriched["Supplier_ID"].tolist())
    row = enriched.loc[enriched["Supplier_ID"] == supplier_id].iloc[0]
    frame = pd.DataFrame([row], columns=[column for column in REQUIRED_COLUMNS if column in row.index])
    label = risk_class(row["Predicted_Risk"])
    probabilities = {column.replace("Probability_", ""): float(row[column]) for column in row.index if column.startswith("Probability_")}
    factors = explain_prediction(pipeline, frame)
    guidance = recommendations(label, row)
    st.write(f"Selected supplier: **{supplier_id}** — predicted risk **{label}**")
    try:
        pdf = build_supplier_pdf(row, label, probabilities, factors, guidance, selected_model)
        st.download_button("Download PDF assessment", pdf, f"supplyguard_{supplier_id}.pdf", "application/pdf", type="primary")
    except RuntimeError as exc:
        st.error(str(exc))
    st.download_button("Download current prediction dataset", enriched.to_csv(index=False).encode("utf-8"), "supplyguard_predictions.csv", "text/csv")
    st.caption("PDF reports include the assessment date, model output, metrics, factors, recommendations, and synthetic-data limitations.")


def page_model_lab(bundle: dict[str, Any], pipeline: Any, selected_model: str) -> None:
    hero("Model laboratory", "Inspect actual saved evaluation artifacts and the selected model's associated features. Metrics are never hardcoded in the interface.", "Model Lab")
    st.warning("The benchmark is synthetic. Evaluation describes performance on constructed labels, not validated real-world supplier outcomes. Probabilities are not guaranteed calibrated confidence values.")
    results = bundle["results"]
    metrics = pd.DataFrame(results).T
    st.subheader("Model comparison")
    visible = [column for column in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "training_seconds"] if column in metrics.columns]
    st.dataframe(metrics[visible].style.format({column: "{:.3f}" for column in visible}), use_container_width=True)
    for name, result in results.items():
        with st.expander(f"{name} evaluation details", expanded=name == selected_model):
            st.write(f"Confusion matrix class order: {result.get('class_order', CLASS_ORDER)}")
            st.dataframe(pd.DataFrame(result["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER), use_container_width=True)
            report = pd.DataFrame(result["classification_report"]).T
            st.dataframe(report.style.format("{:.3f}"), use_container_width=True)
    st.subheader(f"Global associated features — {selected_model}")
    importance = pd.DataFrame(feature_importance(pipeline, top_n=15))
    if not importance.empty:
        st.bar_chart(importance.set_index("feature"), color="#10b981")
        st.dataframe(importance, use_container_width=True, hide_index=True)
    st.caption("Global importance helps describe the model overall; it is not a guaranteed explanation of an individual supplier.")


def main() -> None:
    try:
        data = dataset()
        bundle = model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Install dependencies, then run `python -m src.train_model` to create the saved model artifact.")
        return
    except Exception as exc:
        st.error(f"SupplyGuard could not load the project data or model: {exc}")
        return
    pipeline = bundle["pipeline"]
    selected_model = bundle["selected_model"]
    enriched = predict_dataset(data, pipeline)
    st.sidebar.markdown("<div class='eyebrow'>SupplyGuard AI</div><h2>Procurement intelligence</h2>", unsafe_allow_html=True)
    st.sidebar.caption("Synthetic benchmark decision support")
    pages = ["Overview", "Suppliers", "Assessments", "Supplier Comparison", "Risk Intelligence", "Reports", "Model Lab"]
    page = st.sidebar.radio("Workspace", pages, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.metric("Suppliers", f"{len(data):,}")
    st.sidebar.metric("Selected model", selected_model)
    st.sidebar.warning("Synthetic data only — not real-world validation.")
    if page == "Overview": page_overview(enriched, bundle, selected_model)
    elif page == "Suppliers": page_suppliers(enriched)
    elif page == "Assessments": page_assessments(data, pipeline, selected_model)
    elif page == "Supplier Comparison": page_comparison(enriched, pipeline, selected_model)
    elif page == "Risk Intelligence": page_intelligence(enriched)
    elif page == "Reports": page_reports(enriched, pipeline, selected_model)
    else: page_model_lab(bundle, pipeline, selected_model)


if __name__ == "__main__":
    main()
