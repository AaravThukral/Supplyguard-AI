# SupplyGuard AI

SupplyGuard AI is an AI-based supplier risk assessment and procurement decision-support system. It classifies suppliers as **Low**, **Medium**, or **High** risk using a reproducible scikit-learn pipeline and provides model-associated factors plus transparent procurement guidance.

## Important scientific limitation

The included 1,000-row dataset is synthetic. `Risk_Label` is generated from a project-defined risk score with noise and percentile thresholds. The reported metrics therefore measure performance on this constructed benchmark; they do **not** establish effectiveness or accuracy on real supplier outcomes. Model explanations describe associations, not causes, and predicted probabilities are not guaranteed calibrated confidence values.

## Quick start

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.inspect_dataset
python -m src.train_model
streamlit run app.py
```

The trainer writes `models/supplyguard_model.joblib`, `research/model_results/model_comparison.md`, and `research/model_results/metrics.json`. The model artifact is intentionally generated locally rather than committed to source control.

## Project structure

- `data/suppliers.csv` — local synthetic benchmark.
- `src/generate_dataset.py` — deterministic data generator (`SEED = 42`).
- `src/supplyguard_ml.py` — reusable preprocessing, model, evaluation, explanation, and recommendation utilities.
- `src/train_model.py` — trains and compares Logistic Regression, Decision Tree, and Random Forest using the same stratified split.
- `app.py` — Streamlit dashboard with individual assessment, batch CSV assessment, model information, and limitations.
- `tests/` — automated regression tests.
- `research/model_results/` — historical baseline and generated experiment reports.
- `docs/` — dataset, problem, architecture, roadmap, and research documentation.

## Reproducibility and leakage controls

The target and `Supplier_ID` are excluded from predictors. Numeric scaling and categorical one-hot encoding are fitted inside a `ColumnTransformer` within a scikit-learn `Pipeline`, after the stratified train/test split. Unknown categories are ignored at inference time. The same random state is used for generation and evaluation.

## Testing

```bash
pytest -q
```

The original Logistic Regression report in `research/model_results/logistic_regression.md` is preserved as a historical record. The current comparison report is generated from actual execution of the current pipeline and should be regenerated after changing dependencies or data.
