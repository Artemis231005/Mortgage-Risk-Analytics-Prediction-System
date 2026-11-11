import streamlit as st
import plotly.express as px
from utils import inject_styles, page_header, metric_card, load_csv, ANOMALY_RESULTS

st.set_page_config(page_title="Anomaly Detection | Mortgage Risk Analytics",layout="wide")
inject_styles()
page_header("", "Anomaly detection", "Review unusual financial observations that are flagged.")

iqr=load_csv(str(ANOMALY_RESULTS/"IQR_Results.csv"))
iforest=load_csv(str(ANOMALY_RESULTS/"Isolation_Forest_Results.csv"))

flagged=iforest[iforest["Isolation_Forest_Anomaly"]==True]
rate=len(flagged)/len(iforest) if len(iforest) else 0
c1,c2,c3=st.columns(3)

with c1: metric_card("Records evaluated",f"{len(iforest):,}","Isolation Forest output")
with c2: metric_card("Flagged observations",f"{len(flagged):,}","Model anomaly flag")
with c3: metric_card("Flag rate",f"{rate:.2%}","Approximate configured contamination")

tab1,tab2=st.tabs(["IQR analysis","Isolation Forest"])
with tab1:
    st.markdown("### Interquartile range flags")
    st.write("The IQR method flags values outside the feature-specific lower and upper bounds.")
    
    fig=px.bar(iqr.sort_values("Anomaly_Percentage",ascending=False),x="Feature",y="Anomaly_Percentage",color_discrete_sequence=["#52796F"])
    fig.update_layout(height=360,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="",yaxis_title="Flagged observations (%)")
    fig.update_xaxes(tickangle=-25)
    
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(iqr.style.format({"Q1":"{:,.2f}","Q3":"{:,.2f}","IQR":"{:,.2f}","Lower_Bound":"{:,.2f}","Upper_Bound":"{:,.2f}","Anomaly_Percentage":"{:.2f}%"}),use_container_width=True,hide_index=True)
    st.download_button("Download IQR results",iqr.to_csv(index=False),"IQR_Results.csv","text/csv")

with tab2:
    st.markdown("### Isolation Forest observations")
    st.write("Isolation Forest was configured to flag approximately 1% of the sample.")
    c1,c2=st.columns(2)
    
    with c1:
        counts=iforest["Isolation_Forest_Anomaly"].value_counts().rename(index={True:"Flagged",False:"Not flagged"}).rename_axis("Status").reset_index(name="Records")
        fig=px.pie(counts,names="Status",values="Records",hole=.62,color="Status",color_discrete_map={"Flagged":"#C49A59","Not flagged":"#52796F"})
        fig.update_layout(height=330,margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)
    
    with c2:
        fig=px.histogram(iforest,x="Anomaly_Score",color="Isolation_Forest_Anomaly",nbins=45,
                         color_discrete_map={True:"#C49A59",False:"#52796F"})
        fig.update_layout(height=330,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="Anomaly score",yaxis_title="Records")
        st.plotly_chart(fig,use_container_width=True)
    
    st.markdown("### Filter flagged observations")
    states=["All"]+sorted(iforest["state_code"].dropna().astype(str).unique().tolist())
    state=st.selectbox("State",states)
    display=flagged.copy()
    
    if state!="All": display=display[display["state_code"].astype(str)==state]
    columns=["_source_row","state_code","msa_md_name","action_taken_label","loan_amount","loan_to_value_ratio","loan_term","income","property_value","interest_rate","Anomaly_Score"]
    
    st.caption(f"Showing {min(len(display),500):,} of {len(display):,} flagged records. A maximum of 500 rows is displayed for responsiveness.")
    st.dataframe(display[columns].head(500),use_container_width=True,hide_index=True)
    st.download_button("Download filtered flagged records",display[columns].to_csv(index=False),"isolation_forest_flagged_records.csv","text/csv")
