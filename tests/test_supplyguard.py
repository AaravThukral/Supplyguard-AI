import pandas as pd

from src.generate_dataset import generate_dataset
from src.supplyguard_ml import REQUIRED_COLUMNS, build_pipeline, model_catalog, recommendations, train_all


def test_generator_is_reproducible_and_balanced():
    first = generate_dataset()
    second = generate_dataset()
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (1000, 19)
    assert set(first["Risk_Label"]) == {"Low", "Medium", "High"}


def test_pipeline_fits_preprocessing_inside_training_pipeline():
    data = generate_dataset(123, 120)
    fitted, results, *_ = train_all(data)
    assert set(fitted) == set(model_catalog())
    assert all(0 <= metrics["accuracy"] <= 1 for metrics in results.values())
    assert all("Supplier_ID" not in name for name in fitted["Logistic Regression"].named_steps["preprocessor"].get_feature_names_out())


def test_prediction_handles_unknown_categories():
    data = generate_dataset(123, 120)
    fitted, *_ = train_all(data)
    row = data.iloc[[0]][REQUIRED_COLUMNS].copy()
    row.loc[:, "Region"] = "UnknownRegion"
    prediction = fitted["Logistic Regression"].predict(row)
    assert prediction[0] in {"Low", "Medium", "High"}


def test_recommendations_are_data_linked():
    row = {"On_Time_Delivery_Percentage": 70, "Defect_Rate_Percentage": 9, "Credit_Score": 550, "Supply_Disruption_History": 5}
    output = recommendations("High", row)
    assert any("delivery" in line.lower() for line in output)
    assert any("defect" in line.lower() for line in output)
