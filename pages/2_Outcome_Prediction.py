import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    inject_styles, page_header, metric_card, load_models, load_csv,
    PREDICTION_RESULTS, RESULTS
)

st.set_page_config(page_title="Outcome Prediction | Mortgage Risk Analytics", layout="wide")
inject_styles()
page_header("", "Mortgage outcome prediction", "Use the saved Predictive pipelines to estimate one of the three selected application outcomes, or explore the held-out evaluation results.")


def class_metrics_from_matrix(confusion_matrix, class_names):
    matrix = confusion_matrix.to_numpy(dtype=float)
    rows = []

    for index, class_name in enumerate(class_names):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

        rows.append({
            "Class": class_name,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "Support": int(matrix[index, :].sum())
        })

    return pd.DataFrame(rows)


models = load_models()
model_1 = models["model_1_three_class"]
model_2 = models["model_2_conversion"]
pipeline_1 = model_1["pipeline"]
pipeline_2 = model_2["pipeline"]
features = model_1["feature_columns"]

train_path = RESULTS / "Prepared_Datasets" / "Train.csv"
@st.cache_data(show_spinner=False)
def get_input_options(path):
    data = pd.read_csv(path, low_memory=False)
    return data

train = get_input_options(str(train_path))
categorical_features = [c for c in features if c not in [
    "loan_amount","loan_to_value_ratio","income","property_value",
    "property_value_exempt","tract_to_msa_income_percentage"
]]

with st.expander("Prediction scope",expanded=False):
    st.markdown("""
    - **Model 1** estimates the recorded outcome among Originated, Approved Not Accepted and Denied.
    - **Model 2** compares Originated with Approved Not Accepted within the two-outcome conversion analysis.
    """)

st.markdown("### Enter application characteristics")
st.caption("Enter values in HMDA units. Loan amount and property value are in US dollars; applicant income is in thousands of US dollars.")
with st.form("mortgage_prediction_form"):
    n1,n2,n3 = st.columns(3)
    with n1:
        loan_amount=st.number_input("Loan amount ($)",min_value=0.0,value=250000.0,step=10000.0)
        income=st.number_input("Applicant income ($ thousands)",min_value=0.0,value=100.0,step=5.0)
        property_value=st.number_input("Property value ($)",min_value=0.0,value=350000.0,step=10000.0)
        ltv=st.number_input("Loan-to-value ratio (%)",min_value=0.0,max_value=300.0,value=75.0,step=1.0)
        tract_income=st.number_input("Census tract income (% of MSA median)",min_value=0.0,value=100.0,step=5.0)
        property_exempt=st.selectbox("Property value exemption indicator",options=[0,1],format_func=lambda x:"Not exempt" if x==0 else "Exempt")
    
    with n2:
        loan_type=st.selectbox("Loan type",sorted(train["loan_type_label"].dropna().astype(str).unique()))
        loan_purpose=st.selectbox("Loan purpose",sorted(train["loan_purpose_label"].dropna().astype(str).unique()))
        occupancy=st.selectbox("Occupancy type",sorted(train["occupancy_type_label"].dropna().astype(str).unique()))
        conforming=st.selectbox("Conforming loan limit",sorted(train["conforming_loan_limit"].fillna("Unknown").astype(str).unique()))
        dti=st.selectbox("Debt-to-income ratio",sorted(train["debt_to_income_ratio"].fillna("Unknown").astype(str).unique()))
        preapproval=st.selectbox("Preapproval",sorted(train["preapproval"].fillna("Unknown").astype(str).unique()))
    
    with n3:
        reverse=st.selectbox("Reverse mortgage",sorted(train["reverse_mortgage"].fillna("Unknown").astype(str).unique()))
        open_end=st.selectbox("Open-end line of credit",sorted(train["open-end_line_of_credit"].fillna("Unknown").astype(str).unique()))
        business=st.selectbox("Business or commercial purpose",sorted(train["business_or_commercial_purpose"].fillna("Unknown").astype(str).unique()))
        construction=st.selectbox("Construction method",sorted(train["construction_method"].fillna("Unknown").astype(str).unique()))
        units=st.selectbox("Total units",sorted(train["total_units"].fillna("Unknown").astype(str).unique()))
        state=st.selectbox("State",sorted(train["state_code"].fillna("Unknown").astype(str).unique()))
        msa_options=["Unknown"]+sorted([x for x in train["msa_md_name"].dropna().astype(str).unique() if x!="Unknown"])
        msa=st.selectbox("Metropolitan area",msa_options,index=0)
    submitted=st.form_submit_button("Run outcome models",use_container_width=True)

if submitted:
    record={
        "loan_amount":loan_amount,
        "loan_to_value_ratio":ltv,
        "income":income,
        "property_value":property_value,
        "property_value_exempt":property_exempt,
        "tract_to_msa_income_percentage":tract_income,
        "loan_type_label":loan_type,
        "loan_purpose_label":loan_purpose,
        "occupancy_type_label":occupancy,
        "conforming_loan_limit":conforming,
        "debt_to_income_ratio":dti,
        "preapproval":preapproval,
        "reverse_mortgage":reverse,
        "open-end_line_of_credit":open_end,
        "business_or_commercial_purpose":business,
        "construction_method":construction,
        "total_units":units,
        "state_code":state,
        "msa_md_name":msa
    }
    input_df=pd.DataFrame([record],columns=features)
    
    try:
        probabilities=pipeline_1.predict_proba(input_df)[0]
        if probabilities[1] >= float(model_1["best_threshold"]):
            pred_encoded=1
        elif probabilities[0] >= probabilities[2]:
            pred_encoded=0
        else:
            pred_encoded=2
        predicted_class=model_1["reverse_mapping"][pred_encoded]
        outcome_names={1:"Originated",2:"Approved Not Accepted",3:"Denied"}
        st.markdown("### Model 1 result · Three-Class Mortgage Outcome Prediction")
        st.markdown(f"**Estimated outcome: {outcome_names[predicted_class]}**")
        result_cols=st.columns(3)
        
        for col,cls in zip(result_cols,[1,2,3]):
            with col:
                metric_card(outcome_names[cls],f"{probabilities[cls-1]:.1%}","Model probability")
        st.markdown("### Model 2 result · Mortgage Approval Conversion Analysis")
        
        if predicted_class in [1, 2]:
            conversion_prob=pipeline_2.predict_proba(input_df)[0][1]
            conversion_label="Approved Not Accepted" if conversion_prob >= float(model_2["best_threshold"]) else "Originated"
            st.markdown(f"**Separate conversion-model class: {conversion_label}**")
            st.write(f"Estimated probability of the Approved Not Accepted class: **{conversion_prob:.1%}**")
            st.markdown("An analysis of historical outcomes among approved applications. It does not predict an applicant's willingness or intention to accept a mortgage offer.")
        else:
            st.info("The conversion model is not shown because its training population excludes Denied applications.")
    except Exception as error:
        st.error(f"Prediction could not be generated: {error}")

st.markdown("---")
st.markdown("## Model evaluation")
st.write("The following results are from the saved held-out test evaluation.")

eval_tab1,eval_tab2=st.tabs(["Three-class outcome model","Approval conversion model"])

with eval_tab1:
    comp=load_csv(str(PREDICTION_RESULTS/"Model_Comparison_Optimized.csv"))
    cm=load_csv(str(PREDICTION_RESULTS/"Confusion_Matrix_Optimized.csv"))
    fi=load_csv(str(PREDICTION_RESULTS/"model_1_three_class_Feature_Importance.csv"))
    c1,c2,c3,c4=st.columns(4)
    
    for col,label,val in zip([c1,c2,c3,c4],["Test accuracy","Test Macro F1","Balanced accuracy","Selected threshold"],["85.52%","60.33%","58.61%","0.275"]):
        with col: metric_card(label,val,"Held-out test / selected configuration")
    
    st.markdown("### Validation model comparison")
    st.dataframe(comp.style.format({c:"{:.3f}" for c in comp.select_dtypes("number").columns}),use_container_width=True,hide_index=True)
    st.markdown("### Held-out test confusion matrix")
    
    cm_plot=cm.set_index(cm.columns[0])
    fig=px.imshow(cm_plot,text_auto=True,aspect="auto",color_continuous_scale=["#F5F4EF","#52796F"])
    fig.update_layout(height=380,margin=dict(l=10,r=10,t=25,b=10),xaxis_title="Predicted outcome",yaxis_title="Actual outcome",coloraxis_showscale=False)
    
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Class-level test metrics")
    class_report=class_metrics_from_matrix(cm_plot,["Originated","Approved Not Accepted","Denied"])
    
    st.dataframe(class_report.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-score":"{:.3f}"}),use_container_width=True,hide_index=True)
    st.markdown("### Feature importance")
    
    fi_plot=fi.sort_values("Permutation_Importance",ascending=True).tail(15)
    fig=px.bar(fi_plot,x="Permutation_Importance",y="Feature",orientation="h",color_discrete_sequence=["#52796F"])
    fig.update_layout(height=500,margin=dict(l=10,r=10,t=15,b=10),xaxis_title="Permutation importance",yaxis_title="")
    st.plotly_chart(fig,use_container_width=True)

with eval_tab2:
    comp=load_csv(str(PREDICTION_RESULTS/"Model_2_Conversion_Comparison.csv"))
    cm=load_csv(str(PREDICTION_RESULTS/"Model_2_Conversion_Confusion_Matrix.csv"))
    tradeoff=load_csv(str(PREDICTION_RESULTS/"Model_2_Threshold_Tradeoff.csv"))
    fi=load_csv(str(PREDICTION_RESULTS/"model_2_conversion_Feature_Importance.csv"))
    c1,c2,c3,c4=st.columns(4)
    
    for col,label,val in zip([c1,c2,c3,c4],["Test accuracy","Test Macro F1","Balanced accuracy","Selected threshold"],["93.19%","60.09%","59.33%","0.250"]):
        with col: metric_card(label,val,"Held-out test / selected configuration")
    
    st.markdown("### Validation model comparison")
    st.dataframe(comp.style.format({c:"{:.3f}" for c in comp.select_dtypes("number").columns}),use_container_width=True,hide_index=True)
    st.markdown("### Held-out test confusion matrix")
    
    cm_plot=cm.set_index(cm.columns[0])
    fig=px.imshow(cm_plot,text_auto=True,aspect="auto",color_continuous_scale=["#F5F4EF","#52796F"])
    fig.update_layout(height=340,margin=dict(l=10,r=10,t=25,b=10),xaxis_title="Predicted outcome",yaxis_title="Actual outcome",coloraxis_showscale=False)
    
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Class-level test metrics")
    
    class_report=class_metrics_from_matrix(cm_plot,["Originated","Approved Not Accepted"])
    st.dataframe(class_report.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-score":"{:.3f}"}),use_container_width=True,hide_index=True)
    st.markdown("### Threshold trade-off")
    fig=go.Figure()
    
    for metric in ["Precision","Recall","F1","Specificity"]:
        fig.add_trace(go.Scatter(x=tradeoff["Threshold"],y=tradeoff[metric],mode="lines",name=metric))
    fig.update_layout(height=370,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="Classification threshold",yaxis_title="Metric",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Feature importance")
    fi_plot=fi.sort_values("Permutation_Importance",ascending=True).tail(15)
    fig=px.bar(fi_plot,x="Permutation_Importance",y="Feature",orientation="h",color_discrete_sequence=["#52796F"])
    fig.update_layout(height=500,margin=dict(l=10,r=10,t=15,b=10),xaxis_title="Permutation importance",yaxis_title="")
    
    st.plotly_chart(fig,use_container_width=True)
    st.caption("The high accuracy reflects strong class imbalance. Macro F1 and class-level recall provide a more balanced view of model performance.")
