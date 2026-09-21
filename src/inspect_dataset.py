import pandas as pd

# Load the dataset
data = pd.read_csv("data/suppliers.csv")

print("Dataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nData Types:")
print(data.dtypes)

print("\nBasic Statistics:")
print(data.describe())

print("\nRisk Distribution:")
print(data["Risk_Label"].value_counts())