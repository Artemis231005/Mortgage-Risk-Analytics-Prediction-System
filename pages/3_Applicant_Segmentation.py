import streamlit as st
import plotly.express as px
from utils import inject_styles, page_header, metric_card, load_csv, CLUSTER_RESULTS

st.set_page_config(page_title="Applicant Segmentation | Mortgage Risk Analytics",layout="wide")
inject_styles()
page_header("", "Applicant segmentation", "Explore groups of similarity formed from mortgage and property characteristics.")

kmeans=load_csv(str(CLUSTER_RESULTS/"kmeans_evaluation.csv"))
hier=load_csv(str(CLUSTER_RESULTS/"hierarchical_evaluation.csv"))
numeric=load_csv(str(CLUSTER_RESULTS/"numeric_cluster_profiles.csv"))
categorical=load_csv(str(CLUSTER_RESULTS/"categorical_cluster_profiles.csv"))
outcomes=load_csv(str(CLUSTER_RESULTS/"cluster_outcome_distribution.csv"))

c1,c2,c3=st.columns(3)
with c1: metric_card("Selected K-Means clusters","2","Primary segmentation approach")
with c2: metric_card("K-Means silhouette","0.323","Evaluation sample")
with c3: metric_card("Input features","16","5 numeric · 11 categorical")

tab1,tab2,tab3=st.tabs(["K-Means", "Hierarchical clustering", "Cluster profiles"])
with tab1:
    st.markdown("### K-Means model selection")
    st.write("K-Means was selected as the primary segmentation method. The selected solution uses two clusters, based on the strongest silhouette score among evaluated values of K and the minimum-cluster-size constraint.")
    c1,c2=st.columns(2)
    
    with c1:
        fig=px.line(kmeans,x="k",y="inertia",markers=True,color_discrete_sequence=["#52796F"])
        fig.update_layout(height=330,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="Number of clusters (K)",yaxis_title="Inertia")
        st.plotly_chart(fig,use_container_width=True)
    
    with c2:
        fig=px.line(kmeans,x="k",y="silhouette_score",markers=True,color_discrete_sequence=["#C49A59"])
        fig.update_layout(height=330,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="Number of clusters (K)",yaxis_title="Silhouette score")
        st.plotly_chart(fig,use_container_width=True)
    st.dataframe(kmeans.style.format({c:"{:.3f}" for c in kmeans.select_dtypes("number").columns}),use_container_width=True,hide_index=True)
    st.markdown("### Cluster distribution and outcomes")
    c1,c2=st.columns(2)
    
    with c1:
        st.image(str(CLUSTER_RESULTS/"cluster_distribution.png"),use_container_width=True)
    with c2:
        st.image(str(CLUSTER_RESULTS/"cluster_outcome_distribution.png"),use_container_width=True)

with tab2:
    st.markdown("### Hierarchical clustering evaluation")
    st.write("Hierarchical clustering was evaluated on a smaller sample.")
    fig=px.line(hier,x="k",y="silhouette_score",markers=True,color_discrete_sequence=["#52796F"])
    fig.update_layout(height=330,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="Number of clusters (K)",yaxis_title="Silhouette score")
    
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(hier.style.format({c:"{:.3f}" for c in hier.select_dtypes("number").columns}),use_container_width=True,hide_index=True)
    p=CLUSTER_RESULTS/"hierarchical_dendrogram.png"
    if p.exists(): st.image(str(p),caption="Hierarchical clustering dendrogram (sample)",use_container_width=True)

with tab3:
    st.markdown("### Numerical cluster profiles")
    st.write("Values below are cluster-level averages.")
    st.dataframe(numeric.style.format({c:"{:,.2f}" for c in numeric.select_dtypes("number").columns if c!="kmeans_cluster"}),use_container_width=True,hide_index=True)
    st.markdown("### Categorical cluster profiles")
    st.write("Categorical profiles show the most frequent category for each selected variable within each cluster.")
    st.dataframe(categorical,use_container_width=True,hide_index=True)
    st.markdown("### Application outcome distribution")
    st.dataframe(outcomes,use_container_width=True,hide_index=True)
    st.markdown("### Feature-level comparisons")
    
    chart_names=[
        ("cluster_income.png","Applicant income"),
        ("cluster_loan_amount.png","Loan amount"),
        ("cluster_property_value.png","Property value"),
        ("cluster_loan_term.png","Loan term"),
        ("cluster_loan_to_value_ratio.png","Loan-to-value ratio"),
    ]
    
    for i in range(0,len(chart_names),2):
        c1,c2=st.columns(2)
        for col,(filename,label) in zip([c1,c2],chart_names[i:i+2]):
            with col:
                p=CLUSTER_RESULTS/filename
                if p.exists(): st.image(str(p),caption=label,use_container_width=True)

