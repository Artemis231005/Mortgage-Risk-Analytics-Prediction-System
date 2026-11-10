
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("Dataset/Processed/HDMA Dataset_Processed.csv", low_memory=False)

os.makedirs("Results/Prepared_Datasets", exist_ok=True)


# ============================================================
# TARGET SELECTION
# ============================================================

ml_df = df[df["action_taken"].isin([1, 2, 3])].copy()

ml_df["action_taken"] = pd.to_numeric(ml_df["action_taken"], errors="coerce")
ml_df = ml_df.dropna(subset=["action_taken"])
ml_df["action_taken"] = ml_df["action_taken"].astype(int)

print("\nTarget Distribution:")
print(ml_df["action_taken"].value_counts().sort_index())

print("\nTarget Distribution (%):")
print((ml_df["action_taken"].value_counts(normalize=True) * 100).sort_index())


# ============================================================
# FEATURE SELECTION
# ============================================================

features = [
    # Loan Characteristics
    "loan_type_label",
    "loan_purpose_label",
    "loan_amount",
    "occupancy_type_label",
    "conforming_loan_limit",

    # Financial Characteristics
    "loan_to_value_ratio",
    "debt_to_income_ratio",
    "income",
    "property_value",
    "property_value_exempt",
    "tract_to_msa_income_percentage",

    # Application Details
    "preapproval",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "construction_method",
    "total_units",

    # Geography
    "state_code",
    "msa_md_name"
]

print("\nSelected Features:", len(features))
print(features)


# ============================================================
# CREATE FINAL ML DATASET
# ============================================================

X = ml_df[features].copy()
y = ml_df["action_taken"].copy()

numeric_features = [
    "loan_amount",
    "loan_term",
    "loan_term_exempt",
    "loan_to_value_ratio",
    "loan_to_value_ratio_exempt",
    "income",
    "property_value",
    "property_value_exempt",
    "tract_to_msa_income_percentage"
]

for col in numeric_features:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

categorical_features = [
    col for col in X.columns if col not in numeric_features
]

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

ml_final_df = X.copy()
ml_final_df["action_taken"] = y

print("\nMissing Values:")
print(ml_final_df.isnull().sum().sort_values(ascending=False))

missing_percentage = (
    ml_final_df.isnull().mean() * 100
).sort_values(ascending=False)

print("\nMissing Value Percentage:")
print(missing_percentage)

# ============================================================
# CREATE TRAIN, VALIDATION, TEST SPLITS
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Splits:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# Save Datasets
train_df = X_train.copy()
train_df["action_taken"] = y_train

val_df = X_val.copy()
val_df["action_taken"] = y_val

test_df = X_test.copy()
test_df["action_taken"] = y_test

ml_final_df.to_csv("Results/Prepared_Datasets/ML_Dataset.csv", index=False)
train_df.to_csv("Results/Prepared_Datasets/Train.csv", index=False)
val_df.to_csv("Results/Prepared_Datasets/Validation.csv", index=False)
test_df.to_csv("Results/Prepared_Datasets/Test.csv", index=False)

print("\nTrain, Validation, Test Datasets saved at: Results/Prepared_Datssets")

