"""Summarize supplier characteristics by synthetic risk label."""
from pathlib import Path
import pandas as pd


def main() -> None:
    data = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "suppliers.csv")
    features = ["On_Time_Delivery_Percentage", "Average_Delay_Days", "Lead_Time_Days", "Defect_Rate_Percentage", "Customer_Rating", "Complaint_Count", "Distance_To_Factory_KM", "Transportation_Cost", "Credit_Score", "Debt_Ratio", "Supply_Disruption_History"]
    print(data.groupby("Risk_Label")[features].mean().round(2).to_string())
    print("\nCategorical distributions:")
    for column in ["Region", "ISO_Certified", "Environmental_Compliance"]:
        print(f"\n{column}\n{data.groupby([\"Risk_Label\", column]).size().unstack(fill_value=0)}")


if __name__ == "__main__":
    main()
