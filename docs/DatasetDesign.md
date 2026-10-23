# Dataset Design

## Purpose

The dataset is a locally generated synthetic benchmark for demonstrating supplier-risk classification without using a prohibited public or Kaggle dataset. It contains 1,000 supplier-like records and 19 columns, including the target `Risk_Label`.

## Features

The predictors represent supplier operating, quality, financial, compliance, logistics, and disruption characteristics. `Supplier_ID` is retained for traceability but is excluded from machine-learning predictors. Categorical fields are `Region`, `ISO_Certified`, and `Environmental_Compliance`; all remaining predictor fields are numeric.

## Label construction

The generator combines delivery delay, defect rate, customer rating, complaints, credit score, debt ratio, disruption history, distance, and random noise into a project-defined risk score. The 33rd and 66th percentiles define Low, Medium, and High labels. This creates a roughly balanced benchmark and intentionally reflects an academic simulation rather than observed business outcomes.

## Reproducibility

`src/generate_dataset.py` uses NumPy's seeded generator with seed 42. It writes the dataset relative to the repository root, so it can be run from any working directory with `python -m src.generate_dataset`.

## Quality review

The inspection script reports shape, duplicates, missing values, types, distributions, and descriptive statistics. Because the labels are derived from several observed features, high classification performance is expected and should not be interpreted as independent real-world validation. Future work should compare against anonymized historical supplier outcomes and assess calibration, drift, fairness, and operational utility.
