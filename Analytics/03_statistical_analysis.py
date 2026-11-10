import os
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, f_oneway

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

DATA_PATH = "Dataset/Processed/HDMA Dataset_Processed.csv"

HYPOTHESIS_PATH = "Results/Statistical_Analysis/Hypothesis_Testing"
ANOMALY_PATH = "Results/Statistical_Analysis/Anomaly_Detection"

os.makedirs(HYPOTHESIS_PATH, exist_ok=True)
os.makedirs(ANOMALY_PATH, exist_ok=True)

RANDOM_STATE = 42
MIN_GROUP_SIZE = 100


# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

df["_source_row"] = np.arange(len(df))
df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")

# Prepare features
numeric_features = [
    'loan_amount',
    'loan_to_value_ratio',
    'loan_term',
    'income',
    'property_value',
    'tract_to_msa_income_percentage',
    'interest_rate',
    'rate_spread',
    'total_loan_costs',
    'origination_charges'
]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

categorical_features = [
    "loan_type_label",
    "loan_purpose_label",
    "occupancy_type_label",
    "conforming_loan_limit",
    "preapproval",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "construction_method",
    "total_units",
    "debt_to_income_ratio",
    "state_code"
]

analysis_df = df[df["action_taken"].isin([1, 2, 3])].copy()
analysis_df["action_taken"] = analysis_df["action_taken"].astype(int)

print("\nApplication Outcome Distribution:")
print(
    analysis_df["action_taken_label"]
    .value_counts()
    .to_string()
)

# ============================================================
# PART 1: DESCRIPTIVE STATISTICS
# ============================================================

# Numerical statistics

if numeric_features:
    descriptive_stats = (
        analysis_df[numeric_features]
        .describe()
        .T
    )

    descriptive_stats["Median"] = (analysis_df[numeric_features].median())
    descriptive_stats["Skewness"] = (analysis_df[numeric_features].skew())
    descriptive_stats["Missing_Count"] = (analysis_df[numeric_features].isna().sum())
    descriptive_stats["Missing_Percentage"] = (analysis_df[numeric_features].isna().mean() * 100)

    descriptive_stats.to_csv(
        os.path.join(
            HYPOTHESIS_PATH,
            "Descriptive_Statistics.csv"
        )
    )

    print("\nNumerical Descriptive Statistics:")
    print(descriptive_stats.to_string())


# Categorical statistics
categorical_stats = []

for col in categorical_features:
    counts = (analysis_df[col].fillna("Unknown").astype(str).value_counts())

    for category, count in counts.items():
        categorical_stats.append({
            "Feature": col,
            "Category": category,
            "Count": count,
            "Percentage": (
                count / len(analysis_df) * 100
            )
        })


categorical_stats_df = pd.DataFrame(categorical_stats)

categorical_stats_df.to_csv(
    os.path.join(
        HYPOTHESIS_PATH,
        "Categorical_Statistics.csv"
    ),
    index=False
)

print("\nCategorical Descriptive Statistics:")
print(categorical_stats_df.to_string())

# ============================================================
# DATA QUALITY
# ============================================================

print("\n" + "=" * 65)
print("DATA QUALITY")
print("=" * 65)

quality_results = []

for col in numeric_features:
    series = analysis_df[col].dropna()

    quality_results.append({
        "Feature": col,
        "Valid_Count": len(series),
        "Missing_Count": analysis_df[col].isna().sum(),
        
        "Missing_Percentage": (
            analysis_df[col].isna().mean() * 100
        ),

        "Negative_Count": int((series < 0).sum()),
        "Zero_Count": int((series == 0).sum()),
        
        "Minimum": series.min(),
        "Maximum": series.max(),

        "1st_Percentile": series.quantile(0.01),
        "25th_Percentile": series.quantile(0.25),
        "Median": series.median(),
        "75th_Percentile": series.quantile(0.75),
        "99th_Percentile": series.quantile(0.99)
    })


quality_df = pd.DataFrame(quality_results)

quality_df.to_csv(
    os.path.join(
        HYPOTHESIS_PATH,
        "Numeric_Data_Quality.csv"
    ),
    index=False
)

print("\nData Quality:")
print(quality_df.to_string(index=False))


# ============================================================
# CHI-SQUARE TEST OF INDEPENDENCE
# ============================================================

print("\n" + "=" * 65)
print("CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 65)

chi_square_results = []

for col in categorical_features:
    contingency_table = pd.crosstab(
        analysis_df[col].fillna("Unknown").astype(str),
        analysis_df["action_taken_label"]
    )

    # Skip features with insufficient categories
    if contingency_table.shape[0] < 2:
        continue

    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    n = contingency_table.to_numpy().sum()

    min_dimension = min(contingency_table.shape) - 1

    cramer_v = np.sqrt(chi2 / (n * min_dimension))

    expected_below_5 = int((expected < 5).sum())
    expected_below_1 = int((expected < 1).sum())
    total_cells = expected.size

    expected_warning = (
        expected_below_5 / total_cells > 0.20
        or expected_below_1 > 0
    )

    chi_square_results.append({
        "Feature": col,
        "Chi_Square": chi2,
        "Degrees_of_Freedom": dof,
        "P_Value": p_value,

        "Cramers_V": cramer_v,

        "Expected_Cells_Below_5": expected_below_5,
        "Expected_Cells_Below_1": expected_below_1,
        "Expected_Frequency_Warning": expected_warning
    })

    print("\nFeature:", col)
    print("Chi-Square:", round(chi2, 4))
    print("P-Value:", f"{p_value:.6g}")
    print("Cramer's V:", round(cramer_v, 4))
    print("Expected Frequency Warning:", expected_warning)


chi_square_df = pd.DataFrame(chi_square_results)

chi_square_df.to_csv(
    os.path.join(
        HYPOTHESIS_PATH,
        "Chi_Square_Results.csv"
    ),
    index=False
)


# ============================================================
# ONE-WAY ANOVA
# ============================================================

print("\n" + "=" * 65)
print("ONE-WAY ANOVA")
print("=" * 65)

anova_results = []

for col in numeric_features:
    groups = []

    for class_code in [1, 2, 3]:
        group = analysis_df.loc[
            analysis_df["action_taken"] == class_code,
            col
        ].dropna()
        groups.append(group)

    if any(len(group) < 2 for group in groups):
        continue

    f_statistic, p_value = f_oneway(*groups)

    # Calculate eta-squared to describe the size of the difference between outcome groups.
    all_values = np.concatenate([
        group.to_numpy() for group in groups
    ])

    grand_mean = np.mean(all_values)

    ss_between = sum(
        len(group) * (group.mean() - grand_mean) ** 2
        for group in groups
    )

    ss_total = np.sum(
        (all_values - grand_mean) ** 2
    )

    if ss_total > 0:
        eta_squared = ss_between / ss_total
    else:
        eta_squared = np.nan

    anova_results.append({
        "Feature": col,
        "F_Statistic": f_statistic,
        "P_Value": p_value,
        "Eta_Squared": eta_squared
    })

    print("\nFeature:", col)
    print("F-Statistic:", round(f_statistic, 4))
    print("P-Value:", f"{p_value:.6g}")
    print("Eta-Squared:", round(eta_squared, 6))


anova_df = pd.DataFrame(anova_results)

anova_df.to_csv(
    os.path.join(
        HYPOTHESIS_PATH,
        "ANOVA_Results.csv"
    ),
    index=False
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 65)
print("ANOMALY DETECTION")
print("=" * 65)

anomaly_features = [
    "loan_amount",
    "loan_to_value_ratio",
    "loan_term",
    "income",
    "property_value",
    "tract_to_msa_income_percentage",
    "interest_rate"
]

anomaly_df = analysis_df[anomaly_features].copy()

# Columns to retain in anomaly reports
report_columns = [
    "_source_row",
    "lei",
    "respondent_name",
    "institution_name",
    "lender_name",
    "state_code",
    "msa_md_name",
    "action_taken",
    "action_taken_label"
]

report_columns += [
    col for col in anomaly_features
    if col not in report_columns
]


# ============================================================
# IQR ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 65)
print("IQR ANOMALY DETECTION")
print("=" * 65)

iqr_results = []

iqr_flags = pd.DataFrame(
    False,
    index=analysis_df.index,
    columns=anomaly_features
)

for col in anomaly_features:
    series = anomaly_df[col].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    flags = ((anomaly_df[col] < lower_bound) | (anomaly_df[col] > upper_bound))
    iqr_flags[col] = flags.fillna(False)

    anomaly_count = int(flags.sum())

    iqr_results.append({
        "Feature": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Anomaly_Count": anomaly_count,
        "Anomaly_Percentage": (
            anomaly_count / len(analysis_df) * 100
        )
    })

    print("\nFeature:", col)
    print("Anomaly Count:", anomaly_count)


iqr_summary = pd.DataFrame(iqr_results)

iqr_summary.to_csv(
    os.path.join(
        ANOMALY_PATH,
        "IQR_Results.csv"
    ),
    index=False
)

# ============================================================
# ISOLATION FOREST
# ============================================================

print("\n" + "=" * 65)
print("ISOLATION FOREST")
print("=" * 65)

iforest_features = [
    col for col in anomaly_features
    if anomaly_df[col].notna().any()
]

if iforest_features:
    iforest_input = anomaly_df[iforest_features].copy()

    imputer = SimpleImputer(strategy="median")
    iforest_input = imputer.fit_transform(iforest_input)

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    isolation_predictions = isolation_forest.fit_predict(
        iforest_input
    )

    isolation_scores = isolation_forest.decision_function(
        iforest_input
    )

    valid_report_columns = [
        col for col in report_columns
        if col in analysis_df.columns
    ]
    isolation_results = analysis_df[valid_report_columns].copy()

    isolation_results["Isolation_Forest_Prediction"] = (isolation_predictions)
    isolation_results["Anomaly_Score"] = isolation_scores
    isolation_results["Isolation_Forest_Anomaly"] = (isolation_predictions == -1)

    isolation_results.to_csv(
        os.path.join(
            ANOMALY_PATH,
            "Isolation_Forest_Results.csv"
        ),
        index=False
    )

    anomaly_count = int((isolation_predictions == -1).sum())
    anomaly_percentage = (anomaly_count / len(analysis_df) * 100)

    print("Isolation Forest Anomalies:", anomaly_count)
    print("Isolation Forest Anomaly Percentage:", round(anomaly_percentage, 2))
    

# Final Results
print("\nFiles Saved in:")
print(HYPOTHESIS_PATH)
print(ANOMALY_PATH)

