import streamlit as st
import streamlit.components.v1 as components
from utils import inject_styles, page_header, EDA_GRAPHS, RESULTS, load_csv

st.set_page_config(page_title="Exploratory Analysis | Mortgage Risk Analytics", layout="wide")
inject_styles()
page_header("", "Exploratory data analysis", "Explore the selected HMDA sample across application outcomes, loan characteristics, applicant attributes and geography. Figures are based on the prepared project dataset.")

tabs = st.tabs(["Loan & application", "Financial profile", "Applicant profile", "Geography", "Denial analysis", "Statistical tests"])

with tabs[0]:
    st.markdown("### Application mix and lending characteristics")
    st.write("Application purpose, loan type, occupancy and purchaser patterns provide context for the outcomes represented in the dataset.")
    c1,c2 = st.columns(2)
    with c1:
        for f in ["Number_of_Applications_By_Loan_Purpose.png","Loan_Purpose_vs_Action_Taken.png","Loan_Type_vs_Action_Taken.png","Loan_Purpose_Across_Occupancy_Types.png"]:
            p=EDA_GRAPHS/f
            if p.exists(): st.image(str(p),caption=f.replace("_"," ").replace(".png",""),use_container_width=True)
    with c2:
        for f in ["Number_of_Applications_Per_Loan_Type.png","Number_of_Applications_By_Occupancy_Type.png","Occupancy_Type_vs_Action_Taken.png","Purchaser_Type_vs_Action_Taken.png","Application_Outcomes_Across_Lien_Status.png"]:
            p=EDA_GRAPHS/f
            if p.exists(): st.image(str(p),caption=f.replace("_"," ").replace(".png",""),use_container_width=True)

with tabs[1]:
    st.markdown("### Financial profile")
    st.write("Review loan size, property value, income, debt-to-income and loan-to-value patterns across the three selected outcomes.")
    files=["Distribution_of_Loan_Amount.png","Distribution_of_Interest_Rate.png","Loan_Amount_Across_Application_Outcomes.png","Property_Value_Across_Application_Outcomes.png","Income_Across_Application_Outcomes.png","Loan_To_Value_Ratio_Across_Application_Outcomes.png","Debt_To_Income_Ratio_Across_Application_Outcomes.png","Census_Tract_Income_Percentage_Across_Application_Outcomes.png","Relationship_Between_DTI_and_LTV.png"]
    
    for i in range(0,len(files),2):
        c1,c2=st.columns(2)
        for col,f in zip([c1,c2],files[i:i+2]):
            with col:
                p=EDA_GRAPHS/f
                if p.exists(): st.image(str(p),caption=f.replace("_"," ").replace(".png",""),use_container_width=True)

with tabs[2]:
    st.markdown("### Applicant characteristics")
    st.write("Demographic distributions and outcome differences observed from given dataset.")
    files=["Applicant_Age_vs_Number_of_Applications.png","Applicant_Age_vs_Action_Taken.png","Denial_Rate_Across_Applicant_Age_Groups.png","Applicant_Race_vs_Number_of_Applications.png","Applicant_Race_vs_Action_Taken.png","Denial_Rate_Across_Derived_Race_Groups.png","Applicant_Sex_vs_Number_of_Applications.png","Applicant_Sex_vs_Action_Taken.png","Denial_Rate_Across_Derived_Sex_Groups.png","Applicant_Credit_Score_Type_vs_Action_Taken.png"]
    
    for i in range(0,len(files),2):
        c1,c2=st.columns(2)
        for col,f in zip([c1,c2],files[i:i+2]):
            with col:
                p=EDA_GRAPHS/f
                if p.exists(): st.image(str(p),caption=f.replace("_"," ").replace(".png",""),use_container_width=True)

with tabs[3]:
    st.markdown("### Geographic distribution")
    st.write("The coverage and distribution for states depend on the number of financial institutions and number of records chosen for the dataset.")
    
    for f in ["Top_15_States_by_Mortgage_Applications.png","Top_15_Metropolitan_Areas_by_Mortgage_Applications.png","Denial_Rate_Across_States.png"]:
        p=EDA_GRAPHS/f
        if p.exists(): st.image(str(p),caption=f.replace("_"," ").replace(".png",""),use_container_width=True)
    
    for f in ["Mortgage_Applications_Across_US_States.html","Mortgage_Denial_Rate_Across_US_States.html"]:
        p=EDA_GRAPHS/f
        if p.exists():
            with st.expander(f.replace("_"," ").replace(".html",""),expanded=False):
                components.html(p.read_text(encoding="utf-8"),height=560,scrolling=True)

with tabs[4]:
    st.markdown("### Denial patterns and reported reasons")
    st.write("Denial-reason analysis for applications with a recorded denial outcome against the available HMDA reason fields.")
    files=["Distribution_of_Primary_Denial_Reasons.png","Distribution_of_All_Reported_Denial_Reasons.png","LTV_Distribution_Across_Primary_Denial_Reasons.png","Primary_Denial_Reasons_Across_Racial_Groups.png","Primary_Denial_Reasons_Across_Sex_Groups.png"]
    
    for i in range(0,len(files),2):
        c1,c2=st.columns(2)
        for col,f in zip([c1,c2],files[i:i+2]):
            with col:
                p=EDA_GRAPHS/f
                if p.exists(): st.image(str(p),caption=f.replace("_"," ").replace(".png",""),use_container_width=True)

with tabs[5]:
    st.markdown("### Statistical analysis")
    st.write("The project includes descriptive statistics, chi-square tests and one-way ANOVA.")
    stat_dir=RESULTS/"Statistical_Analysis"/"Hypothesis_Testing"
    
    for filename, title in [
        ("Descriptive_Statistics.csv","Descriptive statistics"),
        ("Chi_Square_Results.csv","Chi-square tests"),
        ("ANOVA_Results.csv","One-way ANOVA"),
        ("Categorical_Statistics.csv","Categorical summaries"),
        ("Numeric_Data_Quality.csv","Numeric data quality")
    ]:
        p=stat_dir/filename
        if p.exists():
            with st.expander(title,expanded=(filename=="Chi_Square_Results.csv")):
                data=load_csv(str(p))
                st.dataframe(data,use_container_width=True,hide_index=True)
                st.download_button("Download table",data=data.to_csv(index=False),file_name=filename,mime="text/csv",key=filename)
