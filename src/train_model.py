
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data = pd.read_csv("data/suppliers.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

# Separate features and target
X = data.drop(["Risk_Label", "Supplier_ID"], axis=1)
y = data["Risk_Label"]

# Encode categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(X[categorical_columns])

encoded_columns = encoder.get_feature_names_out(categorical_columns)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoded_columns,
    index=X.index
)

# Remove original categorical columns and add encoded columns
X = X.drop(columns=categorical_columns)
X = pd.concat([X, encoded_df], axis=1)

print("\nEncoded feature shape:")
print(X.shape)

print("\nFinal feature columns:")
print(X.columns.tolist())

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

print("\nTarget distribution:")
print(y.value_counts())

print("\nCategorical columns:")
print(X.select_dtypes(include=["object"]).columns.tolist())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())