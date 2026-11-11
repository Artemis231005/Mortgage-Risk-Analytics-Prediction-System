from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results"
EDA_GRAPHS = RESULTS / "EDA_Graphs"
CLUSTER_RESULTS = RESULTS / "ML" / "Clustering_Results"
PREDICTION_RESULTS = RESULTS / "ML" / "Prediction_Results"
ANOMALY_RESULTS = RESULTS / "Statistical_Analysis" / "Anomaly_Detection"

@st.cache_data(show_spinner=False)
def load_processed_data():
    path = ROOT / "Results" / "Prepared_Datasets" / "ML_Dataset.csv"
    data = pd.read_csv(path, low_memory=False)
    data["action_taken_label"] = data["action_taken"].map({
        1: "Originated",
        2: "Approved Not Accepted",
        3: "Denied"
    })
    return data

@st.cache_data(show_spinner=False)
def load_csv(path_string):
    return pd.read_csv(path_string)

@st.cache_resource(show_spinner="Loading trained prediction models...")
def load_models():
    return joblib.load(PREDICTION_RESULTS / "Best_Models.pkl")

def inject_styles():
    css_path = ROOT / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def page_header(kicker, title, description):
    if kicker:
        st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    if description:
        st.markdown(description)

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def show_image(filename, caption=None):
    path = EDA_GRAPHS / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)

def show_result_image(filename, caption=None):
    candidates = list((RESULTS / "ML").rglob(filename))
    if candidates:
        st.image(str(candidates[0]), caption=caption, use_container_width=True)

def format_number(value):
    return f"{value:,.0f}"
