# System Architecture

SupplyGuard AI has four local layers:

1. **Data layer** — `data/suppliers.csv` and the deterministic generator.
2. **ML layer** — `src/supplyguard_ml.py` loads validated columns, splits data, fits a training-only `ColumnTransformer`, trains three classifiers, evaluates them, and serializes the selected pipeline.
3. **Decision-support layer** — model explanations expose global importance or local contribution signals; separate rule-based recommendations use the predicted class and visible input thresholds.
4. **Presentation layer** — `app.py` provides dashboard, individual assessment, batch assessment, and model-information views through Streamlit.

The serialized artifact contains the fitted preprocessing and estimator together, preventing training-time and inference-time transformations from diverging. Batch assessment validates required columns and missing values and marks invalid rows instead of silently dropping them.

The prediction is not the recommendation: the classifier outputs a risk label and, where supported, probabilities; the recommendation layer provides transparent review actions. Neither is a causal guarantee or a substitute for procurement policy and human review.
