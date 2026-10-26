
## Enterprise interface

The Streamlit app now uses a dark navy, emerald, amber, and coral visual system with seven workspace sections:

| Section | Purpose |
|---|---|
| Overview | KPI cards, predicted and synthetic-label distributions, model snapshot, and suppliers deserving attention. |
| Suppliers | Searchable and filterable supplier directory with CSV export. |
| Assessments | Existing-supplier or manual-profile assessments with probabilities, model-associated factors, alerts, guidance, and PDF export. |
| Supplier Comparison | Two- or three-supplier trade-off matrix covering risk, delivery, quality, financial, logistics, and disruption metrics. |
| Risk Intelligence | Transparent threshold alerts and an exportable review-priority list. |
| Reports | Individual PDF assessment and full prediction CSV downloads. |
| Model Lab | Actual saved model metrics, confusion matrices, classification reports, and global feature associations. |

The interface always distinguishes `Risk_Label` (the synthetic dataset label) from `Predicted_Risk` (the saved model output). Rule-based alerts are implemented separately in `src/reporting.py` and are not represented as machine-learning predictions. The local watchlist and alert view are based on the current dataset snapshot; the app does not claim to monitor live suppliers over time.

### Demonstration flow

After training the model, launch `streamlit run app.py`. Start in Overview to show the population KPIs, open Suppliers to filter high-risk records, use Supplier Comparison to discuss trade-offs, then assess one supplier and download its PDF report. Finish in Model Lab to show the actual evaluation artifacts and the synthetic-data warning. For a clean demonstration, use the existing-supplier assessment mode so the inputs are traceable to `data/suppliers.csv`.

PDF generation uses ReportLab and is included in `requirements.txt`. If the optional dependency is unavailable, the Reports page shows an actionable error while CSV exports remain available.
