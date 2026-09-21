
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Scale numerical features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeatures scaled successfully!")

# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# Make predictions on test data
y_pred = model.predict(X_test)

print("\nPredictions generated successfully!")
print("First 10 predictions:")
print(y_pred[:10])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

print("\nConfusion Matrix:")
print(cm)