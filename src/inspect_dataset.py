"""Print structural and quality checks for the synthetic supplier dataset."""
from pathlib import Path
import pandas as pd


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "suppliers.csv"
    data = pd.read_csv(path)
    print(f"Dataset: {path}")
    print(f"Shape: {data.shape}")
    print(f"Duplicate rows: {data.duplicated().sum()}")
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print("\nColumn names:")
    print(data.columns.tolist())
    print("\nData types:")
    print(data.dtypes)
    print("\nRisk distribution:")
    print(data["Risk_Label"].value_counts().sort_index())
    print("\nNumeric summary:")
    print(data.describe().transpose())


if __name__ == "__main__":
    main()
