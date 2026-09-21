import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of suppliers
NUM_SUPPLIERS = 1000

supplier_ids = [
    f"SUP{str(i).zfill(4)}"
    for i in range(1, NUM_SUPPLIERS + 1)
]

years_in_business = np.random.randint(1, 31, NUM_SUPPLIERS)

regions = np.random.choice(
    ["North", "South", "East", "West", "Central"],
    NUM_SUPPLIERS
)

on_time_delivery = np.random.uniform(60, 99, NUM_SUPPLIERS)

average_delay = np.clip(
    15 - (on_time_delivery - 60) * 0.36
    + np.random.normal(0, 1.2, NUM_SUPPLIERS),
    0,
    15
)

defect_rate = np.random.uniform(0.5, 12, NUM_SUPPLIERS)

customer_rating = np.clip(
    5 - defect_rate * 0.15
    + np.random.normal(0, 0.25, NUM_SUPPLIERS),
    1,
    5
)

complaint_count = np.clip(
    (5 - customer_rating) * 8
    + np.random.normal(0, 3, NUM_SUPPLIERS),
    0,
    40
).astype(int)

manufacturing_capacity = np.random.randint(
    5000, 100000, NUM_SUPPLIERS
)

annual_revenue = np.random.uniform(
    10, 1000, NUM_SUPPLIERS
)

credit_score = np.random.randint(
    450, 850, NUM_SUPPLIERS
)

debt_ratio = np.round(
    np.random.uniform(0.1, 0.8, NUM_SUPPLIERS),
    2
)

iso_certified = np.random.choice(
    ["Yes", "No"],
    NUM_SUPPLIERS,
    p=[0.7, 0.3]
)

environmental_compliance = np.random.choice(
    ["Yes", "No"],
    NUM_SUPPLIERS,
    p=[0.75, 0.25]
)

iso_certified = np.random.choice(
    ["Yes", "No"],
    NUM_SUPPLIERS,
    p=[0.7, 0.3]
)

environmental_compliance = np.random.choice(
    ["Yes", "No"],
    NUM_SUPPLIERS,
    p=[0.75, 0.25]
)

distance_to_factory = np.random.uniform(
    10, 2000, NUM_SUPPLIERS
)

transportation_cost = (
    distance_to_factory * np.random.uniform(
        20, 60, NUM_SUPPLIERS
    )
)
supply_disruption_history = np.random.poisson(
    2,
    NUM_SUPPLIERS
)

supply_disruption_history = np.clip(
    supply_disruption_history,
    0,
    8
)
risk_score = (
    (100 - on_time_delivery) * 0.30
    + average_delay * 2.0
    + defect_rate * 2.0
    + (5 - customer_rating) * 8
    + complaint_count * 0.5
    + (850 - credit_score) * 0.05
    + debt_ratio * 10
    + supply_disruption_history * 3
    + distance_to_factory / 500
)

risk_score += np.random.normal(
    0,
    3,
    NUM_SUPPLIERS
)
risk_label = np.where(
    risk_score < 35,
    "Low",
    np.where(
        risk_score < 55,
        "Medium",
        "High"
    )
)
data = pd.DataFrame({
    "Supplier_ID": supplier_ids,
    "Years_In_Business": years_in_business,
    "Region": regions,
    "Manufacturing_Capacity": manufacturing_capacity,
    "On_Time_Delivery_Percentage": np.round(on_time_delivery, 2),
    "Average_Delay_Days": np.round(average_delay, 2),
    "Lead_Time_Days": np.random.randint(2, 30, NUM_SUPPLIERS),
    "Defect_Rate_Percentage": np.round(defect_rate, 2),
    "Customer_Rating": np.round(customer_rating, 2),
    "Complaint_Count": complaint_count,
    "Annual_Revenue": np.round(annual_revenue, 2),
    "Credit_Score": credit_score,
    "Debt_Ratio": debt_ratio,
    "ISO_Certified": iso_certified,
    "Environmental_Compliance": environmental_compliance,
    "Distance_To_Factory_KM": np.round(distance_to_factory, 2),
    "Transportation_Cost": np.round(transportation_cost, 2),
    "Supply_Disruption_History": supply_disruption_history,
    "Risk_Label": risk_label
})
data.to_csv(
    "data/suppliers.csv",
    index=False
)

print("Dataset created successfully!")
print("Number of suppliers:", len(data))
print("\nFirst 5 suppliers:")
print(data.head())
print("\nRisk distribution:")
print(data["Risk_Label"].value_counts())