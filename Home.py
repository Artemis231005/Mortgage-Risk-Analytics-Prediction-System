import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from utils import (
    inject_styles, metric_card, load_processed_data,
)

st.set_page_config(
    page_title="Home | Mortgage Risk Analytics & Prediction System",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_styles()

with st.sidebar:
    st.markdown("## Mortgage Risk Analytics And Prediction System")
    st.markdown("**A clearer view of mortgage lending**")
    st.markdown("---")
    st.caption("ANALYTICS WORKSPACE")
    st.markdown("Use the pages above to explore lending patterns, model outcomes, applicant segments and unusual observations.")
    st.markdown("---")

df = load_processed_data()
outcomes = df["action_taken_label"].value_counts()

@st.cache_data(show_spinner=False)
def load_eda_counts():
    return {
        "total": 115033,
        "originated": 75976,
        "approved_not_accepted": 3879,
        "denied": 13162
    }


eda_counts = load_eda_counts()

total = eda_counts["total"]
originated = eda_counts["originated"]
approved_not_accepted = eda_counts["approved_not_accepted"]
denied = eda_counts["denied"]

st.markdown(
    '<div class="hero"><div class="eyebrow">MORTGAGE MARKET RESEARCH · 2025 HMDA</div>'
    '<h1>A clearer view of mortgage lending</h1>'
    '<p>An analytical view of mortgage application outcomes, approval conversion patterns, applicant segments and unusual lending observations.</p></div>',
    unsafe_allow_html=True
)

c1,c2,c3,c4 = st.columns(4)
with c1: metric_card("Applications analysed", f"{total:,}", "2025 HMDA records")
with c2: metric_card("Originated", f"{originated:,}", f"{originated/total:.1%} of total applications")
with c3: metric_card("Approved · not accepted", f"{approved_not_accepted:,}", f"{approved_not_accepted/total:.1%} of total applications")
with c4: metric_card("Denied", f"{denied:,}", f"{denied/total:.1%} of total applications")

st.markdown("### Application outcome profile")
left,right = st.columns([1.05,1])

with left:
    outcome_df = df[df["action_taken"].isin([1,2,3])]["action_taken_label"].value_counts().rename_axis("Outcome").reset_index(name="Applications")
    fig = px.bar(outcome_df, x="Outcome", y="Applications", color="Outcome", 
                color_discrete_map={"Originated":"#52796F","Approved Not Accepted":"#C49A59","Denied":"#172B3A"})
    fig.update_layout(showlegend=False, height=360, margin=dict(l=10,r=10,t=25,b=10),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#42545F"),yaxis_title="Applications",xaxis_title="")
    fig.update_yaxes(showgrid=True,gridcolor="#E4E6E1")
    st.plotly_chart(fig,use_container_width=True)
    
with right:
    st.markdown("#### What this project covers")
    st.markdown("""
    - **Exploratory analysis** — applicant, loan, financial and geographic patterns.
    - **Outcome prediction** — three-class application outcome modelling.
    - **Approval conversion** — analysis of originated versus approved-not-accepted outcomes.
    - **Segmentation** — K-Means and hierarchical clustering profiles.
    - **Anomaly Detection** - flags on data for further review.
    """)

st.markdown("### Project at a glance")
a,b,c = st.columns(3)

with a:
    st.markdown("**01 / Understand**")
    st.write("Examine distributions, relationships and differences across mortgage applications.")

with b:
    st.markdown("**02 / Predict**")
    st.write("Estimate the recorded application outcome using the ML models.")

with c:
    st.markdown("**03 / Investigate**")
    st.write("Explore applicant segments and observations that differ from common patterns.")

