"""Generate the reproducible synthetic supplier benchmark used by SupplyGuard AI."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
NUM_SUPPLIERS = 1000


def generate_dataset(seed: int = SEED, num_suppliers: int = NUM_SUPPLIERS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    supplier_ids = [f"SUP{i:04d}" for i in range(1, num_suppliers + 1)]
    years = rng.integers(1, 31, num_suppliers)
    regions = rng.choice(["North", "South", "East", "West", "Central"], num_suppliers)
    on_time = rng.uniform(60, 99, num_suppliers)
    average_delay = np.clip(15 - (on_time - 60) * 0.36 + rng.normal(0, 1.2, num_suppliers), 0, 15)
    defect_rate = rng.uniform(0.5, 12, num_suppliers)
    rating = np.clip(5 - defect_rate * 0.15 + rng.normal(0, 0.25, num_suppliers), 1, 5)
    complaints = np.clip((5 - rating) * 8 + rng.normal(0, 3, num_suppliers), 0, 40).astype(int)
    capacity = rng.integers(5000, 100000, num_suppliers)
    revenue = rng.uniform(10, 1000, num_suppliers)
    credit = rng.integers(450, 850, num_suppliers)
    debt = np.round(rng.uniform(0.1, 0.8, num_suppliers), 2)
    iso = rng.choice(["Yes", "No"], num_suppliers, p=[0.7, 0.3])
    environmental = rng.choice(["Yes", "No"], num_suppliers, p=[0.75, 0.25])
    distance = rng.uniform(10, 2000, num_suppliers)
    transportation = distance * rng.uniform(20, 60, num_suppliers)
    disruptions = np.clip(rng.poisson(2, num_suppliers), 0, 8)
    risk_score = ((100 - on_time) * 0.30 + average_delay * 2.0 + defect_rate * 2.0 + (5 - rating) * 8 + complaints * 0.5 + (850 - credit) * 0.05 + debt * 10 + disruptions * 3 + distance / 500 + rng.normal(0, 3, num_suppliers))
    low, high = np.percentile(risk_score, [33, 66])
    labels = np.where(risk_score <= low, "Low", np.where(risk_score <= high, "Medium", "High"))
    return pd.DataFrame({
        "Supplier_ID": supplier_ids, "Years_In_Business": years, "Region": regions,
        "Manufacturing_Capacity": capacity, "On_Time_Delivery_Percentage": np.round(on_time, 2),
        "Average_Delay_Days": np.round(average_delay, 2), "Lead_Time_Days": rng.integers(2, 30, num_suppliers),
        "Defect_Rate_Percentage": np.round(defect_rate, 2), "Customer_Rating": np.round(rating, 2),
        "Complaint_Count": complaints, "Annual_Revenue": np.round(revenue, 2), "Credit_Score": credit,
        "Debt_Ratio": debt, "ISO_Certified": iso, "Environmental_Compliance": environmental,
        "Distance_To_Factory_KM": np.round(distance, 2), "Transportation_Cost": np.round(transportation, 2),
        "Supply_Disruption_History": disruptions, "Risk_Label": labels,
    })


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "suppliers.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    data.to_csv(output, index=False)
    print(f"Dataset written to {output} ({len(data)} rows, {len(data.columns)} columns)")
    print(data["Risk_Label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
