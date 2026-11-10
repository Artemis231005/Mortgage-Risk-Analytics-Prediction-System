
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

df = pd.read_csv("Dataset/Processed/HDMA Dataset_Processed.csv", low_memory=False)
os.makedirs("Results/EDA_Graphs", exist_ok=True)

# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

# Total Number of Mortgage Applications
print("Total Number of Mortgage Applications:")
print(df['lei'].count())


# Total Number of Successful Loans
print("\nTotal Number of Successful Loans:")
print(df[df['action_taken'] == 1]['lei'].count())


# Total Number of Mortgage Applications Approved but Not Accepted
print("\nTotal Number of Mortgage Applications Approved but Not Accepted:")
print(df[df['action_taken'] == 2]['lei'].count())


# Total Number of Mortgage Applications Rejected
print("\nTotal Number of Mortgage Applications Rejected:")
print(df[df['action_taken'] == 3]['lei'].count())


# Observed Year for Mortgage Applications
print("\nObserved Year for Mortgage Applications:")
print(df['activity_year'].value_counts())


# ============================================================
# DATA PREPARATION 
# ============================================================

# Select the three application outcomes used for analysis
eda_df = df[df['action_taken'].isin([1, 2, 3])].copy()

# Get dataset for denied applications
denied_df = df[df['action_taken'] == 3].copy()

denial_reason_cols = [f'denial_reason-{i}' for i in range(1, 5)]


# ============================================================
# GENERAL APPLICATION ANALYSIS
# ============================================================

# Distribution of Interest Rate
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, y='interest_rate')

plt.title("Distribution of Interest Rate")
plt.ylabel("Interest Rate (%)")
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Distribution of Loan Amount
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, y=df['loan_amount'] / 100000)

plt.title("Distribution of Loan Amount")
plt.ylabel("Loan Amount ($100,000s)")
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Number of Applications Per Loan Type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='loan_type_label')

plt.title("Number of Applications Per Loan Type")
plt.ylabel("Number of Applications")
plt.xlabel("Loan Type")
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Number of Applications By Loan Purpose
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='loan_purpose_label')

plt.title("Number of Applications By Loan Purpose")
plt.ylabel("Number of Applications")
plt.xlabel("Loan Purpose")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Number of Applications By Purchaser Type
plt.figure(figsize=(18, 7))
sns.countplot(data=df, x='purchaser_type_label')

plt.title("Number of Applications By Purchaser Type")
plt.ylabel("Number of Applications")
plt.xlabel("Purchaser Type")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Number of Applications By Occupancy Type
plt.figure(figsize=(11, 6))
sns.countplot(data=df, x='occupancy_type_label')

plt.title("Number of Applications By Occupancy Type")
plt.ylabel("Number of Applications")
plt.xlabel("Occupancy Type")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# ============================================================
# APPLICATION OUTCOME ANALYSIS
# ============================================================

# Loan Type vs Action Taken
ct = pd.crosstab(
    eda_df['loan_type_label'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(11, 6))

plt.xlabel("Loan Type")
plt.ylabel("Percentage of Applications")
plt.title("Loan Type vs Action Taken")
plt.legend(title="Action Taken")
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Loan Purpose vs Action Taken
ct = pd.crosstab(
    eda_df['loan_purpose_label'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(13, 7))

plt.xlabel("Loan Purpose")
plt.ylabel("Percentage of Applications")
plt.title("Loan Purpose vs Action Taken")
plt.legend(title="Action Taken")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Purchaser Type vs Action Taken
ct = pd.crosstab(
    eda_df['purchaser_type_label'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(18, 7))

plt.xlabel("Purchaser Type")
plt.ylabel("Percentage of Applications")
plt.title("Purchaser Type vs Action Taken")
plt.legend(title="Action Taken")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Occupancy Type vs Action Taken
ct = pd.crosstab(
    eda_df['occupancy_type_label'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(11, 6))

plt.xlabel("Occupancy Type")
plt.ylabel("Percentage of Applications")
plt.title("Occupancy Type vs Action Taken")
plt.legend(title="Action Taken")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Lien Status vs Action Taken
ct = pd.crosstab(
    eda_df['lien_status_label'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title("Application Outcomes Across Lien Status")
plt.xlabel("Lien Status")
plt.ylabel("Percentage of Applications")
plt.legend(title="Application Outcome")
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Loan Purpose vs Loan Type
ct = pd.crosstab(
    eda_df['loan_purpose_label'],
    eda_df['loan_type_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(14, 7))

plt.title("Loan Type Distribution Across Loan Purposes")
plt.xlabel("Loan Purpose")
plt.ylabel("Percentage of Applications")
plt.legend(title="Loan Type", bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Occupancy Type vs Loan Purpose
ct = pd.crosstab(
    eda_df['occupancy_type_label'],
    eda_df['loan_purpose_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(13, 7))

plt.title("Loan Purpose Across Occupancy Types")
plt.xlabel("Occupancy Type")
plt.ylabel("Percentage of Applications")
plt.legend(title="Loan Purpose", bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# ============================================================
# DEMOGRAPHIC ANALYSIS
# ============================================================


# Applicant Age vs Number of Applications
plt.figure(figsize=(11, 6))
sns.countplot(data=df, x='applicant_age', order=df['applicant_age'].value_counts().index)

plt.title("Applicant Age vs Number of Applications")
plt.xlabel("Applicant Age")
plt.ylabel("Number of Applications")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Applicant Age vs Action Taken
ct = pd.crosstab(
    eda_df['applicant_age'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(11, 6))

plt.xlabel("Applicant Age")
plt.ylabel("Percentage of Applications")
plt.title("Applicant Age vs Action Taken")
plt.legend(title="Action Taken")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Applicant Sex vs Number of Applications
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x='derived_sex')

plt.title("Applicant Sex vs Number of Applications")
plt.xlabel("Applicant Sex")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Applicant Sex vs Action Taken
ct = pd.crosstab(
    eda_df['derived_sex'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100

ct.plot(kind='bar', stacked=True, figsize=(9, 5))

plt.xlabel("Applicant Sex")
plt.ylabel("Percentage of Applications")
plt.title("Applicant Sex vs Action Taken")
plt.legend(title="Action Taken")
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Applicant Race vs Number of Applications
plt.figure(figsize=(16, 7))
sns.countplot(data=df, x='derived_race')

plt.title("Applicant Race vs Number of Applications")
plt.xlabel("Applicant Race")
plt.ylabel("Number of Applications")
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Applicant Race vs Action Taken
ct = pd.crosstab(
    eda_df['derived_race'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100

ct.plot(kind='bar', stacked=True, figsize=(16, 7))

plt.xlabel("Applicant Race")
plt.ylabel("Percentage of Applications")
plt.title("Applicant Race vs Action Taken")
plt.legend(title="Action Taken")
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Applicant Credit Score Type vs Action Taken
ct = pd.crosstab(
    eda_df['applicant_credit_score_type_label'],
    eda_df['action_taken_label'],
    normalize='index'
) * 100

ct.plot(kind='bar', stacked=True, figsize=(16, 7))

plt.xlabel("Applicant Credit Score Type")
plt.ylabel("Percentage of Applications")
plt.title("Applicant Credit Score Type vs Action Taken")
plt.legend(title="Action Taken")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# ============================================================
# DENIAL ANALYSIS
# ============================================================

# Denied Applications vs Primary Denial Reason
plt.figure(figsize=(12, 7))
sns.countplot(data=denied_df, y='denial_reason-1_label', order=denied_df['denial_reason-1_label'].value_counts().index)

plt.xlabel("Number of Denied Applications")
plt.ylabel("Primary Denial Reason")
plt.title("Distribution of Primary Denial Reasons")
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Primary Denial Reason vs Race
ct = pd.crosstab(
    denied_df['derived_race'],
    denied_df['denial_reason-1_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(16, 7))

plt.xlabel("Applicant Race")
plt.ylabel("Percentage of Denied Applications")
plt.title("Primary Denial Reasons Across Racial Groups")
plt.legend(title="Denial Reason", bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Primary Denial Reason vs Sex
ct = pd.crosstab(
    denied_df['derived_sex'],
    denied_df['denial_reason-1_label'],
    normalize='index'
) * 100


ct.plot(kind='bar', stacked=True, figsize=(11, 6))

plt.xlabel("Applicant Sex")
plt.ylabel("Percentage of Denied Applications")
plt.title("Primary Denial Reasons Across Sex Groups")
plt.legend(title="Denial Reason", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Distribution of All Reported Denial Reasons
all_reasons = denied_df[
    [col + '_label' for col in denial_reason_cols]
].melt(value_name='Denial Reason')

all_reasons = all_reasons.dropna()
all_reasons = all_reasons[all_reasons['Denial Reason'] != 'Not Applicable']


plt.figure(figsize=(13, 7))
sns.countplot(data=all_reasons, y='Denial Reason',order=all_reasons['Denial Reason'].value_counts().index)

plt.xlabel("Number of Reported Reasons")
plt.ylabel("Denial Reason")
plt.title("Distribution of All Reported Denial Reasons")
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Loan-to-Value Ratio Across Primary Denial Reasons
denied_ltv_df = denied_df.dropna(subset=['loan_to_value_ratio', 'denial_reason-1_label']).copy()
denied_ltv_df['loan_to_value_ratio'] = pd.to_numeric(denied_ltv_df['loan_to_value_ratio'], errors='coerce')

denied_ltv_df = denied_ltv_df.dropna(subset=['loan_to_value_ratio'])
ltv_upper = denied_ltv_df['loan_to_value_ratio'].quantile(0.99)

plt.figure(figsize=(16, 7))
sns.boxplot(data=denied_ltv_df, x='denial_reason-1_label', y='loan_to_value_ratio', showfliers=False)

plt.ylim(0, ltv_upper)
plt.xlabel("Primary Denial Reason")
plt.ylabel("Loan-to-Value Ratio")
plt.title("LTV Distribution Across Primary Denial Reasons")
plt.xticks(rotation=40, ha='right')
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()

# Denial Rate Across Demographic Groups
denial_rate_cols = [
    'derived_race',
    'derived_sex',
    'applicant_age'
]


for col in denial_rate_cols:
    denial_rate = (
        eda_df.groupby(col)['is_denied']
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(x=denial_rate.index, y=denial_rate.values)

    plt.title(f"Denial Rate Across {col.replace('_', ' ').title()} Groups")
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel("Denial Rate (%)")
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------
# FINANCIAL AND APPLICATION OUTCOMES ANALYSIS 
# ------------------------------------------------------------

# Financial Characteristics Across Application Outcomes
financial_eda_cols = [
    'loan_to_value_ratio',
    'debt_to_income_ratio',
    'income',
    'loan_amount',
    'property_value'
]

for col in financial_eda_cols:
    plot_df = eda_df.dropna(subset=[col]).copy()

    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df = plot_df.dropna(subset=[col])

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=plot_df, x='action_taken_label', y=col, showfliers=False)

    if col in ['loan_to_value_ratio', 'income', 'loan_amount', 'property_value']:
        plt.ylim(0, plot_df[col].quantile(0.99))

    plt.title(f"{col.replace('_', ' ').title()} Across Application Outcomes")
    plt.xlabel("Application Outcome")
    plt.ylabel(col.replace('_', ' ').title())
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

# DTI vs LTV
dti_ltv_df = eda_df.copy()
dti_ltv_df = dti_ltv_df.dropna(subset=['debt_to_income_ratio', 'loan_to_value_ratio'])


plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=dti_ltv_df.sample(
        min(5000, len(dti_ltv_df)),
        random_state=42
    ),
    x='debt_to_income_ratio',
    y='loan_to_value_ratio',
    hue='action_taken_label',
    alpha=0.5
)

plt.title("Relationship Between DTI and LTV")
plt.xlabel("Debt-to-Income Ratio")
plt.ylabel("Loan-to-Value Ratio")
plt.legend(title="Application Outcome")
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Census Tract Outcome vs Application Outcome
tract_df = eda_df.dropna(subset=['tract_to_msa_income_percentage'])

plt.figure(figsize=(11, 6))
sns.boxplot(data=tract_df, x='action_taken_label', y='tract_to_msa_income_percentage')

plt.title("Census Tract Income Percentage Across Application Outcomes")
plt.xlabel("Application Outcome")
plt.ylabel("Tract Income as Percentage of MSA Median")
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# ============================================================
# 6. GEOGRAPHICAL ANALYSIS
# ============================================================


# Top 15 States By Mortgage Applications
top_states = df['state_code'].value_counts().head(15)

plt.figure(figsize=(11, 7))
sns.barplot(x=top_states.values, y=top_states.index)

plt.title("Top 15 States by Mortgage Applications")
plt.xlabel("Number of Applications")
plt.ylabel("State")
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# State-wise Denial Rate
state_denial = (eda_df.groupby('state_code')['is_denied'].agg(['mean', 'count']))

state_denial['denial_rate'] = state_denial['mean'] * 100
state_denial = state_denial[state_denial['count'] >= 30]

state_denial = state_denial.sort_values('denial_rate', ascending=False)


plt.figure(figsize=(15, 7))
sns.barplot(x=state_denial.index, y=state_denial['denial_rate'])

plt.title("Denial Rate Across States")
plt.xlabel("State")
plt.ylabel("Denial Rate (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()


# Top 15 Metropolitan Areas by Mortgage Applications
top_msa = df['msa_md_name'].value_counts().head(15)

plt.figure(figsize=(13, 7))
sns.barplot(x=top_msa.values, y=top_msa.index)

plt.title("Top 15 Metropolitan Areas by Mortgage Applications")
plt.xlabel("Number of Applications")
plt.ylabel("Metropolitan Area")
plt.savefig(f"EDA_Graphs/{plt.gcf().axes[0].get_title().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
plt.show()

# Total Applications By State
state_applications = (
    df.groupby('state_code')
    .size()
    .reset_index(name='application_count')
)


fig = px.choropleth(
    state_applications,
    locations='state_code',
    locationmode='USA-states',
    color='application_count',
    scope='usa',
    color_continuous_scale='Blues',
    hover_name='state_code',
    hover_data={
        'application_count': True
    },
    title='Mortgage Applications Across US States'
)

fig.update_geos(fitbounds='locations', visible=False)
fig.write_html("EDA_Graphs/Mortgage_Applications_Across_US_States.html")
fig.show()


# State-Wise Denial Rate
state_denial_map = (
    eda_df.groupby('state_code')
    .agg(
        total_applications=('is_denied', 'count'),
        denied_applications=('is_denied', 'sum')
    )
    .reset_index()
)


state_denial_map['denial_rate'] = (
    state_denial_map['denied_applications'] /
    state_denial_map['total_applications'] * 100
)

state_denial_map = state_denial_map[
    state_denial_map['total_applications'] >= 30
].copy()


fig = px.choropleth(
    state_denial_map,
    locations='state_code',
    locationmode='USA-states',
    color='denial_rate',
    scope='usa',
    color_continuous_scale='Reds',
    hover_name='state_code',
    hover_data={
        'total_applications': True,
        'denied_applications': True,
        'denial_rate': ':.2f'
    },
    title='Mortgage Denial Rate Across US States (%)'
)

fig.update_geos(fitbounds='locations', visible=False)
fig.write_html("EDA_Graphs/Mortgage_Denial_Rate_Across_US_States.html")
fig.show()
