"""Pure helpers for transparent risk intelligence and supplier reports."""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

import pandas as pd


ALERT_RULES = [
    ("Delivery reliability", "On_Time_Delivery_Percentage", lambda value: float(value) < 85, "On-time delivery is below 85%.", "High"),
    ("Quality", "Defect_Rate_Percentage", lambda value: float(value) > 8, "Defect rate is above 8%.", "High"),
    ("Disruption history", "Supply_Disruption_History", lambda value: int(value) >= 4, "Supply disruption history is elevated.", "High"),
    ("Credit", "Credit_Score", lambda value: float(value) < 600, "Credit score is below 600.", "Medium"),
    ("Customer experience", "Complaint_Count", lambda value: int(value) >= 15, "Complaint count is 15 or higher.", "Medium"),
    ("Leverage", "Debt_Ratio", lambda value: float(value) > 0.60, "Debt ratio is above 0.60.", "Medium"),
]


def alert_records(row: pd.Series | dict[str, Any]) -> list[dict[str, str]]:
    values = row.to_dict() if isinstance(row, pd.Series) else row
    alerts = []
    for name, field, condition, message, severity in ALERT_RULES:
        try:
            triggered = condition(values.get(field))
        except (TypeError, ValueError):
            triggered = False
        if triggered:
            alerts.append({"category": name, "field": field, "message": message, "severity": severity})
    return alerts


def alert_summary(row: pd.Series | dict[str, Any]) -> str:
    alerts = alert_records(row)
    if not alerts:
        return "No rule-based alert thresholds triggered."
    return "; ".join(alert["message"] for alert in alerts)


def build_supplier_pdf(
    row: pd.Series | dict[str, Any],
    label: str,
    probabilities: dict[str, float] | None,
    factors: list[dict[str, Any]],
    guidance: list[str],
    selected_model: str,
    generated_at: datetime | None = None,
) -> bytes:
    """Build a compact PDF report. ReportLab is imported lazily for optional CLI use."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires reportlab. Install dependencies from requirements.txt.") from exc

    values = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    output = __import__("io").BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#0f172a")
    styles["Heading2"].textColor = colors.HexColor("#047857")
    story = [Paragraph("SupplyGuard AI — Supplier Assessment", styles["Title"]), Spacer(1, 0.12 * inch)]
    supplier_id = escape(str(values.get("Supplier_ID", "Ad hoc assessment")))
    assessment_date = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"Supplier: {supplier_id}<br/>Assessment date: {assessment_date}<br/>Selected model: {escape(selected_model)}", styles["BodyText"]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph(f"Predicted risk category: <b>{escape(label)}</b>", styles["Heading2"]))
    if probabilities:
        probability_rows = [["Risk class", "Model output"]] + [[escape(name), f"{float(value):.1%}"] for name, value in probabilities.items()]
        table = Table(probability_rows, colWidths=[2.2 * inch, 1.4 * inch])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d1fae5")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")), ("PADDING", (0, 0), (-1, -1), 6)]))
        story.extend([table, Spacer(1, 0.18 * inch)])
    story.append(Paragraph("Supplier metrics", styles["Heading2"]))
    fields = ["On_Time_Delivery_Percentage", "Average_Delay_Days", "Lead_Time_Days", "Defect_Rate_Percentage", "Customer_Rating", "Credit_Score", "Debt_Ratio", "Supply_Disruption_History", "Transportation_Cost"]
    metric_rows = [["Metric", "Value"]] + [[field.replace("_", " "), escape(str(values.get(field, "—")))] for field in fields]
    metric_table = Table(metric_rows, colWidths=[3.1 * inch, 2.2 * inch])
    metric_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")), ("PADDING", (0, 0), (-1, -1), 5)]))
    story.extend([metric_table, Spacer(1, 0.18 * inch), Paragraph("Model-associated factors", styles["Heading2"])])
    story.extend([Paragraph(f"{escape(str(item.get('feature')))}: {float(item.get('contribution', 0)):.3f}", styles["BodyText"]) for item in factors] or [Paragraph("No local factor explanation was available.", styles["BodyText"])])
    story.extend([Spacer(1, 0.12 * inch), Paragraph("Procurement guidance", styles["Heading2"])])
    story.extend([Paragraph(f"• {escape(item)}", styles["BodyText"]) for item in guidance])
    story.extend([Spacer(1, 0.12 * inch), Paragraph("Limitations", styles["Heading2"]), Paragraph("This assessment uses a synthetic benchmark whose labels are generated from a project-defined risk score. Model outputs describe associations, not causes, and are not validated real-world supplier outcomes or guaranteed calibrated confidence values.", styles["BodyText"])])
    document.build(story)
    return output.getvalue()


def build_batch_pdf(rows: pd.DataFrame, selected_model: str, generated_at: datetime | None = None) -> bytes:
    """Build a compact multi-supplier summary PDF for a validated batch."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires reportlab. Install dependencies from requirements.txt.") from exc

    output = __import__("io").BytesIO()
    document = SimpleDocTemplate(output, pagesize=landscape(letter), rightMargin=0.4 * inch, leftMargin=0.4 * inch, topMargin=0.4 * inch, bottomMargin=0.4 * inch)
    styles = getSampleStyleSheet()
    rows = rows.copy()
    probability_columns = [column for column in rows.columns if column.startswith("Probability_")]
    columns = [column for column in ["Supplier_ID", "Predicted_Risk", "On_Time_Delivery_Percentage", "Defect_Rate_Percentage", "Credit_Score", "Supply_Disruption_History"] if column in rows.columns] + probability_columns
    header = [column.replace("Probability_", "P(").replace("_", " ") + (")" if column in probability_columns else "") for column in columns]
    values = [header]
    for _, row in rows.iterrows():
        values.append([f"{float(row[column]):.1%}" if column in probability_columns else str(row.get(column, "—")) for column in columns])
    table = Table(values, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d1fae5")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")), ("FONTSIZE", (0, 0), (-1, -1), 7), ("PADDING", (0, 0), (-1, -1), 4)]))
    generated = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    story = [Paragraph("SupplyGuard AI — Batch Assessment Summary", styles["Title"]), Paragraph(f"Generated: {generated}<br/>Selected model: {selected_model}<br/>Rows included: {len(rows)}", styles["BodyText"]), Spacer(1, 0.15 * inch), table, Spacer(1, 0.15 * inch), Paragraph("This batch report summarizes model outputs from a synthetic benchmark. Outputs are not validated real-world supplier outcomes, and threshold alerts should be reviewed separately.", styles["BodyText"])]
    document.build(story)
    return output.getvalue()
