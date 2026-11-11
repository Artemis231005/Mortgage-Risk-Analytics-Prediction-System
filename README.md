# Mortgage Risk Analytics & Prediction System

A Streamlit application for exploring selected 2025 HMDA mortgage application records, modelling application outcomes, examining approval conversion patterns, segmenting applications and investigating unusual observations.

## Run locally

Use Python 3.11 or 3.12.

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the application
streamlit run Home.py
```

## Application pages

1. **Overview** — dataset summary and selected application outcome profile.
2. **Exploratory Analysis** — loan, financial, applicant, geographic and statistical analysis.
3. **Outcome Prediction** — interactive predictions and evaluation for the three-class outcome and approval-conversion models.
4. **Applicant Segmentation** — K-Means and hierarchical clustering evaluation and cluster profiles.
5. **Anomaly Detection** — IQR and Isolation Forest results with record-level review.
6. **About** — project scope, methodology and limitations.

## Project structure

```text
Mortgage Risk Analytics & Prediction System/
├── Home.py
├── utils.py
├── pages/
├── assets/
├── .streamlit/
├── Analytics/
├── ML/
├── Dataset/
│   ├── Raw/
│   └── Processed/
├── Results/
│   ├── EDA_Graphs/
│   ├── ML/
│   ├── Prepared_Datasets/
│   └── Statistical_Analysis/
└── Report/
```

## Notes

- The application reuses the saved trained pipelines in `Results/ML/Prediction_Results/Best_Models.pkl`.
- The model page expects the trained model artifact and prepared training data to remain in their existing relative paths.
- The dataset is a selected subset of 2025 HMDA reporting institutions (115,033 total mortgage application records in the processed EDA dataset), not the complete national dataset.
- Model outputs are analytical estimates, not underwriting recommendations or credit decisions.
- The conversion model classifies recorded outcomes among originated and approved-not-accepted applications. It does not estimate an applicant's willingness to accept a mortgage.
- Clusters are descriptive and anomaly flags require review; neither should be treated as individual risk grades.

## Deployment

Deploy the repository to Streamlit Community Cloud:
1. Push the project to GitHub.
2. Select `Home.py` as the application entry point.
3. Keep `requirements.txt`, `Results/Prepared_Datasets/ML_Dataset.csv`, `Results/Prepared_Datasets/Train.csv`, `Results/ML/Prediction_Results/Best_Models.pkl`, the prediction evaluation CSVs, clustering CSVs, anomaly CSVs and the existing graph/report assets in the repository. The raw and full processed HMDA source CSVs are not required by the deployed app.
4. Set the Python version to one compatible with the pinned dependencies.

The repository includes the processed dataset and analysis outputs used by the app. For public deployment, review HMDA data redistribution terms and repository size before publishing.
