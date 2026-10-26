import pandas as pd

from src.reporting import alert_records, alert_summary


def test_alert_rules_are_transparent_and_data_linked():
    row = pd.Series({
        "On_Time_Delivery_Percentage": 70,
        "Defect_Rate_Percentage": 9,
        "Supply_Disruption_History": 5,
        "Credit_Score": 550,
        "Complaint_Count": 2,
        "Debt_Ratio": 0.2,
    })
    alerts = alert_records(row)
    categories = {alert["category"] for alert in alerts}
    assert {"Delivery reliability", "Quality", "Disruption history", "Credit"}.issubset(categories)
    assert "On-time delivery is below 85%." in alert_summary(row)


def test_alert_rules_do_not_flag_healthy_values():
    row = {
        "On_Time_Delivery_Percentage": 96,
        "Defect_Rate_Percentage": 1,
        "Supply_Disruption_History": 0,
        "Credit_Score": 760,
        "Complaint_Count": 2,
        "Debt_Ratio": 0.25,
    }
    assert alert_records(row) == []
