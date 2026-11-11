import streamlit as st
from utils import inject_styles, page_header

st.set_page_config(page_title="About | Mortgage Risk Analytics",layout="wide")
inject_styles()
page_header("", "About this project",
            "A research-oriented mortgage analytics application built on a selected subset of the 2025 Home Mortgage Disclosure Act dataset.")

st.markdown("## Project objectives")
st.markdown("""
1. Explore mortgage application outcomes and lending patterns across selected institutions.
2. Model three application outcomes: Originated, Approved Not Accepted and Denied.
3. Examine the conversion outcome among applications recorded as Originated or Approved Not Accepted.
4. Segment applications using mortgage, applicant-income and property characteristics.
5. Identify unusual observations using statistical and machine-learning methods.
""")

st.markdown("## Data and scope")
st.markdown("""
- **Source:** 2025 HMDA loan application data.
- **Coverage:** A selected set of reporting institutions, not the complete national HMDA population.
- **Total mortgage applications analysed:** 115,033 records.
- **Primary outcome classes:** Originated (1), Approved Not Accepted (2) and Denied (3).
- **Use:** Exploratory research and model demonstration.
""")

st.markdown("## Important limitations")
st.markdown("""
- The institution-filtered sample limits how far findings can be generalized.
- Class imbalance affects the interpretation of accuracy, particularly for the conversion model.
- Demographic distributions and outcome differences are descriptive and should be interpreted carefully, not as causal effects or stand-alone lending criteria.
- Geographic and institution-level variables may encode structural differences and should be interpreted carefully.
""")

st.markdown("## Technology")
st.markdown("Python · Pandas · NumPy · Scikit-learn · LightGBM · Plotly · Streamlit")
st.markdown("---")
st.caption("Mortgage Risk Analytics & Prediction System · 2025 HMDA selected-institution sample")
