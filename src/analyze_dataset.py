import pandas as pd

data = pd.read_csv("data/suppliers.csv")

print("Average values by Risk Label:")
print(
    data.groupby("Risk_Label")[
        [
            "On_Time_Delivery_Percentage",
            "Average_Delay_Days",
            "Lead_Time_Days",
            "Defect_Rate_Percentage",
            "Customer_Rating",
            "Complaint_Count",
            "Distance_To_Factory_KM",
            "Supply_Disruption_History"
        ]
    ].mean().round(2)
)

print("\nCategorical values:")

print("\nISO Certified:")
print(data["ISO_Certified"].value_counts())

print("\nEnvironmental Compliance:")
print(data["Environmental_Compliance"].value_counts())

print("\nRegion:")
print(data["Region"].value_counts())

print("\nRisk Profile:")

risk_features = [
    "On_Time_Delivery_Percentage",
    "Average_Delay_Days",
    "Lead_Time_Days",
    "Defect_Rate_Percentage",
    "Customer_Rating",
    "Complaint_Count",
    "Distance_To_Factory_KM",
    "Transportation_Cost"
]

print(
    data.groupby("Risk_Label")[risk_features]
    .mean()
    .round(2)
)