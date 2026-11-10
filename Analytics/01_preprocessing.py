import numpy as np
import pandas as pd

# Load Data
df = pd.read_csv("Dataset/Raw/HDMA Dataset.csv", low_memory=False)

# Check Duplicates
print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Check Misisng Values
# print("\nMissing Values")
# for col in df.columns:
#     print(f"{col}: {df[col].isnull().sum()}")

impute_list = [
    'county_code',
    'state_code',
    'census_tract',
    'conforming_loan_limit'
]

for col in impute_list:
    df[col] = df[col].fillna("Unknown")

# Handle exempt category 
df['loan_term_exempt'] = df['loan_term'].eq('Exempt').astype(int)

df['interest_rate_exempt'] = df['interest_rate'].eq('Exempt').astype(int)

df["debt_to_income_ratio"] = (
    df["debt_to_income_ratio"]
    .fillna("Unknown")
    .astype(str)
)

df['loan_to_value_ratio_exempt'] = df['loan_to_value_ratio'].eq('Exempt').astype(int)

df['property_value_exempt'] = df['property_value'].eq('Exempt').astype(int)

# Handle 8888 category in applicant_age
df['applicant_age'] = df['applicant_age'].replace(['8888', 8888], 'Unknown')

# Handle total_units
df['total_units'] = df['total_units'].fillna('Unknown').astype(str)

# Get Application Outcome Labels
action_taken_map = {
    1: 'Originated',
    2: 'Approved but Not Accepted',
    3: 'Denied'
}

df['action_taken_label'] = (
    pd.to_numeric(df['action_taken'], errors='coerce')
    .map(action_taken_map)
)

df['is_denied'] = (pd.to_numeric(df['action_taken'], errors='coerce') == 3).astype(int)


# Get Loan Type Labels
loan_type_map = {
    1: 'Conventional',
    2: 'FHA-insured',
    3: 'VA-guaranteed',
    4: 'USDA / FSA-guaranteed'
}

df['loan_type_label'] = (
    pd.to_numeric(df['loan_type'], errors='coerce')
    .map(loan_type_map)
)

# Get Loan Purpose Labels
loan_purpose_map = {
    1: 'Home Purchase',
    2: 'Home Improvement',
    31: 'Refinancing',
    32: 'Cash-out Refinancing',
    4: 'Other Purpose',
    5: 'Not Applicable'
}

df['loan_purpose_label'] = (
    pd.to_numeric(df['loan_purpose'], errors='coerce')
    .map(loan_purpose_map)
)

# Get Purchaser Type Labels
purchaser_type_map = {
    0: 'Not Applicable',
    1: 'Fannie Mae',
    2: 'Ginnie Mae',
    3: 'Freddie Mac',
    4: 'Farmer Mac',
    5: 'Private Securitizer',
    6: 'Commercial Bank',
    71: 'Credit Union/Finance Company',
    72: 'Life Insurance Company',
    8: 'Affiliate Institution',
    9: 'Other Purchaser'
}

df['purchaser_type_label'] = (
    pd.to_numeric(df['purchaser_type'], errors='coerce')
    .map(purchaser_type_map)
)

# Get Occupancy Type Labels
occupancy_type_map = {
    1: 'Principal Residence',
    2: 'Second Residence',
    3: 'Investment Property'
}

df['occupancy_type_label'] = (
    pd.to_numeric(df['occupancy_type'], errors='coerce')
    .map(occupancy_type_map)
)

# Get Credit Score Labels
credit_score_map = {
    1: 'Equifax Beacon 5.0',
    2: 'Experian Fair Isaac',
    3: 'FICO Risk Score Classic 04',
    4: 'FICO Risk Score Classic 98',
    5: 'VantageScore 2.0',
    6: 'VantageScore 3.0',
    7: 'More than one credit scoring model',
    8: 'Other credit scoring model',
    9: 'Not applicable',
    10: 'No co-applicant',
    11: 'FICO Score 9',
    12: 'FICO Score 8',
    13: 'FICO Score 10',
    14: 'FICO Score 10T',
    15: 'VantageScore 4.0',
    1111: 'Exempt'
}

df['applicant_credit_score_type_label'] = (
    pd.to_numeric(
        df['applicant_credit_score_type'],
        errors='coerce'
    ).map(credit_score_map)
)   

# Get Lien Status Labels
lien_status_mapping = {
    1: "First Lien",
    2: "Subordinate Lien"
}

df["lien_status_label"] = df["lien_status"].map(lien_status_mapping)
df["lien_status_label"] = df["lien_status_label"].fillna("Unknown")

# Get MSA/MD Codes
msa_df = pd.read_excel("Dataset/Raw/MSA_Codes_Dataset.xlsx", header=2)

msa_mapping_df = msa_df.iloc[:, :2].copy()
msa_mapping_df.columns = ['derived_msa-md', 'msa_md_name']

msa_mapping_df = msa_mapping_df.dropna(subset=['derived_msa-md', 'msa_md_name'])

msa_mapping_df['derived_msa-md'] = (
    pd.to_numeric(msa_mapping_df['derived_msa-md'], errors='coerce')
    .astype('Int64')
    .astype('string')
)

msa_mapping_df = msa_mapping_df.dropna(subset=['derived_msa-md'])

msa_mapping = dict(zip(
    msa_mapping_df['derived_msa-md'],
    msa_mapping_df['msa_md_name']
))

df['derived_msa-md'] = (
    pd.to_numeric(df['derived_msa-md'], errors='coerce')
    .astype('Int64')
    .astype('string')
)

df['msa_md_name'] = df['derived_msa-md'].map(msa_mapping)
df['msa_md_name'] = df['msa_md_name'].fillna('Unknown')

# Get Denial Reason Labels
denial_reason_map = {
    1: 'Debt-to-Income Ratio',
    2: 'Employment History',
    3: 'Credit History',
    4: 'Collateral',
    5: 'Insufficient Cash',
    6: 'Unverifiable Information',
    7: 'Credit Application Incomplete',
    8: 'Mortgage Insurance Denied',
    9: 'Other',
    10: 'Not Applicable'
}

denial_reason_cols = [
    f'denial_reason-{i}' for i in range(1, 5)
]

for col in denial_reason_cols:
    df[col + '_label'] = (
        pd.to_numeric(df[col], errors='coerce')
        .map(denial_reason_map)
    )


# Handle numeric
financial_cols = [
    'loan_to_value_ratio',
    'income',
    'loan_amount',
    'property_value',
    'interest_rate',
    'tract_to_msa_income_percentage'
]

for col in financial_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
# Save Processed Dataset

output_path = "Dataset/Processed/HDMA Dataset_Processed.csv"
df.to_csv(output_path, index=False)

print(f"\nProcessed dataset saved successfully at: {output_path}")
