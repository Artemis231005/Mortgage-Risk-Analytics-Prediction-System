import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster.hierarchy import linkage, dendrogram


INPUT_PATH = "Dataset/Processed/HDMA Dataset_Processed.csv"
OUTPUT_DIR = "Results/ML/Clustering_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

MIN_K = 2
MAX_K = 8

EVALUATION_SAMPLE = 10000
SILHOUETTE_SAMPLE = 3000

HIERARCHICAL_SAMPLE = 2500
DENDROGRAM_SAMPLE = 500

LOWER_PERCENTILE = 0.01
UPPER_PERCENTILE = 0.99

MIN_CLUSTER_PERCENTAGE = 1.0


# Load Dataset
print("=" * 65)
print("MORTGAGE CLUSTERING AND SEGMENTATION")
print("=" * 65)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\nDataset Shape:", df.shape)


# Prepare Features

numeric_features = [
    "loan_amount",
    "income",
    "property_value",
    "loan_term",
    "loan_to_value_ratio",
]

categorical_features = [
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "lien_status",
    "total_units",
    "construction_method",
    "derived_dwelling_category",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "debt_to_income_ratio"
]


# Remove categorical columns with excessive unique values
categorical_features = [
    col for col in categorical_features
    if 0 < df[col].nunique(dropna=True) <= 50
]

# ============================================================
# PREPARE CLUSTERING DATA
# ============================================================

cluster_df = df[numeric_features + categorical_features].copy()

for col in numeric_features:
    cluster_df[col] = pd.to_numeric(
        cluster_df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )
    
    cluster_df.loc[cluster_df[col] < 0, col] = np.nan


for col in categorical_features:
    cluster_df[col] = cluster_df[col].astype("string")
    cluster_df[col] = cluster_df[col].replace(["", "nan", "None"], pd.NA)

cluster_df = cluster_df[
    numeric_features + categorical_features
]

# Remove records where all selected features are missing
cluster_df = cluster_df.dropna(how="all")

# print("\nClustering Dataset Shape:", cluster_df.shape)
# print("Records Retained:", len(cluster_df))


# ============================================================
# OUTLIER HANDLING
# ============================================================

model_df = cluster_df.copy()

print("\nNumerical Outlier Handling:")

for col in numeric_features:
    lower_limit = model_df[col].quantile(LOWER_PERCENTILE)
    upper_limit = model_df[col].quantile(UPPER_PERCENTILE)

    outlier_count = ((model_df[col] < lower_limit) | (model_df[col] > upper_limit)).sum()

    model_df[col] = model_df[col].clip(lower=lower_limit, upper=upper_limit)


# ============================================================
# LOG TRANSFORMATION
# ============================================================

log_features = [
    "loan_amount",
    "income",
    "property_value"
]

for col in log_features:
        model_df[col] = np.log1p(model_df[col])


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

X = preprocessor.fit_transform(model_df)


# ============================================================
# K-MEANS CLUSTER EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("K-MEANS CLUSTERING")
print("=" * 65)

np.random.seed(RANDOM_STATE)

evaluation_size = min(EVALUATION_SAMPLE, X.shape[0])

evaluation_indices = np.random.choice(
    X.shape[0],
    size=evaluation_size,
    replace=False
)

X_evaluation = X[evaluation_indices]

# Dense data is required for Davies-Bouldin and Calinski-Harabasz scores
if hasattr(X_evaluation, "toarray"):
    X_evaluation_dense = X_evaluation.toarray()
else:
    X_evaluation_dense = X_evaluation


kmeans_results = []

for k in range(MIN_K, MAX_K + 1):
    print("\nEvaluating K =", k)

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(X_evaluation)

    sil = silhouette_score(
        X_evaluation,
        labels,
        sample_size=min(SILHOUETTE_SAMPLE, evaluation_size),
        random_state=RANDOM_STATE
    )

    db = davies_bouldin_score(
        X_evaluation_dense,
        labels
    )

    ch = calinski_harabasz_score(
        X_evaluation_dense,
        labels
    )

    cluster_counts = np.bincount(labels)
    min_cluster_pct = (cluster_counts.min() / evaluation_size) * 100

    kmeans_results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": sil,
        "davies_bouldin_score": db,
        "calinski_harabasz_score": ch,
        "min_cluster_percentage": min_cluster_pct
    })

    print("Inertia:", model.inertia_)
    print("Silhouette Score:", sil)
    print("Davies-Bouldin Score:", db)
    print("Calinski-Harabasz Score:", ch)
    print("Smallest Cluster (%):", round(min_cluster_pct, 3))


kmeans_results_df = pd.DataFrame(kmeans_results)

kmeans_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "kmeans_evaluation.csv"),
    index=False
)


# ============================================================
# K-MEANS EVALUATION PLOTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(
    kmeans_results_df["k"],
    kmeans_results_df["inertia"],
    marker="o"
)

axes[0].set_title("K-Means Elbow Method")
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("Inertia")

axes[1].plot(kmeans_results_df["k"], kmeans_results_df["silhouette_score"], marker="o")

axes[1].set_title("Silhouette Score vs K")
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("Silhouette Score")

plt.savefig(
    os.path.join(OUTPUT_DIR, "kmeans_evaluation.png"),
    dpi=300
)
plt.close()


# ============================================================
# SELECT K-MEANS CLUSTER COUNT
# ============================================================

valid_results = kmeans_results_df[kmeans_results_df["min_cluster_percentage"] >= MIN_CLUSTER_PERCENTAGE]

if valid_results.empty:
    valid_results = kmeans_results_df

optimal_k = int(
    valid_results.loc[valid_results["silhouette_score"].idxmax(), "k"]
)

print("\nSelected K-Means Cluster Count:", optimal_k)


# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

kmeans_final = KMeans(
    n_clusters=optimal_k,
    random_state=RANDOM_STATE,
    n_init=10
)

kmeans_labels = kmeans_final.fit_predict(X)

cluster_df["kmeans_cluster"] = kmeans_labels

print("\nK-Means Cluster Distribution:")
print(cluster_df["kmeans_cluster"].value_counts().sort_index())

final_cluster_pct = (cluster_df["kmeans_cluster"].value_counts(normalize=True).sort_index() * 100)

print("\nK-Means Cluster Percentages:")
print(final_cluster_pct.round(3))


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

print("\n" + "=" * 65)
print("HIERARCHICAL CLUSTERING")
print("=" * 65)

hierarchical_size = min(HIERARCHICAL_SAMPLE, X.shape[0])

hierarchical_indices = np.random.choice(
    X.shape[0],
    size=hierarchical_size,
    replace=False
)

X_hierarchical = X[hierarchical_indices]

if hasattr(X_hierarchical, "toarray"):
    X_hierarchical = X_hierarchical.toarray()

print("\nHierarchical Sample Size:", hierarchical_size)


# ============================================================
# HIERARCHICAL DENDROGRAM
# ============================================================

dendrogram_size = min(DENDROGRAM_SAMPLE, hierarchical_size)

dendrogram_indices = np.random.choice(
    hierarchical_size,
    size=dendrogram_size,
    replace=False
)

X_dendrogram = X_hierarchical[dendrogram_indices]

linkage_matrix = linkage(X_dendrogram, method="ward")

plt.figure(figsize=(15, 7))
 
dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Merged Clusters")
plt.ylabel("Ward Distance")

plt.savefig(
    os.path.join(OUTPUT_DIR, "hierarchical_dendrogram.png"),
    dpi=300
)
plt.close()


# ============================================================
# HIERARCHICAL CLUSTER EVALUATION
# ============================================================

hierarchical_results = []

for k in range(MIN_K, min(MAX_K, hierarchical_size - 1) + 1):
    print("\nEvaluating Hierarchical K =", k)

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(X_hierarchical)

    sil = silhouette_score(
        X_hierarchical,
        labels,
        sample_size=min(SILHOUETTE_SAMPLE, hierarchical_size),
        random_state=RANDOM_STATE
    )

    db = davies_bouldin_score(
        X_hierarchical,
        labels
    )

    ch = calinski_harabasz_score(
        X_hierarchical,
        labels
    )

    cluster_counts = np.bincount(labels)

    min_cluster_pct = (cluster_counts.min() / hierarchical_size) * 100

    hierarchical_results.append({
        "k": k,
        "silhouette_score": sil,
        "davies_bouldin_score": db,
        "calinski_harabasz_score": ch,
        "min_cluster_percentage": min_cluster_pct
    })

    print("Silhouette Score:", sil)
    print("Davies-Bouldin Score:", db)
    print("Calinski-Harabasz Score:", ch)
    print("Smallest Cluster (%):", round(min_cluster_pct, 3))


hierarchical_results_df = pd.DataFrame(hierarchical_results)

hierarchical_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "hierarchical_evaluation.csv"),
    index=False
)


# ============================================================
# SELECT HIERARCHICAL CLUSTER COUNT
# ============================================================

valid_results = hierarchical_results_df[hierarchical_results_df["min_cluster_percentage"] >= MIN_CLUSTER_PERCENTAGE]

if valid_results.empty:
    valid_results = hierarchical_results_df

optimal_hierarchical_k = int(
    valid_results.loc[valid_results["silhouette_score"].idxmax(), "k"]
)

print("\nSelected Hierarchical K:", optimal_hierarchical_k)


# ============================================================
# FINAL HIERARCHICAL MODEL
# ============================================================

hierarchical_final = AgglomerativeClustering(n_clusters=optimal_hierarchical_k, linkage="ward")

hierarchical_labels = hierarchical_final.fit_predict(X_hierarchical)

hierarchical_df = cluster_df.iloc[hierarchical_indices].copy()
hierarchical_df["hierarchical_cluster"] = hierarchical_labels

print("\nHierarchical Cluster Distribution:")
print(hierarchical_df["hierarchical_cluster"].value_counts().sort_index())

hierarchical_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hierarchical_cluster_assignments.csv"
    ),
    index=True
)


# ============================================================
# NUMERICAL CLUSTER PROFILES
# ============================================================

print("\n" + "=" * 65)
print("K-MEANS CLUSTER INTERPRETATION")
print("=" * 65)

if numeric_features:
    numeric_profile = cluster_df.groupby(
        "kmeans_cluster"
    )[numeric_features].median()

    print("\nNumerical Feature Medians:")
    print(numeric_profile)

    numeric_profile.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "numeric_cluster_profiles.csv"
        )
    )


# ============================================================
# CATEGORICAL CLUSTER PROFILES
# ============================================================

categorical_profile = cluster_df.groupby(
    "kmeans_cluster"
)[categorical_features].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
)

print("\nCategorical Feature Modes:")
print(categorical_profile)

categorical_profile.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "categorical_cluster_profiles.csv"
    )
)


# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

cluster_sizes = (cluster_df["kmeans_cluster"] .value_counts() .sort_index())

cluster_distribution = pd.DataFrame({
    "cluster": cluster_sizes.index,
    "count": cluster_sizes.values,
    "percentage": cluster_sizes.values / len(cluster_df) * 100
})

print("\nCluster Distribution:")
print(cluster_distribution)

plt.figure(figsize=(9, 5))

sns.barplot(data=cluster_distribution, x="cluster", y="percentage")

plt.title("Mortgage Applicant Segments")
plt.xlabel("Cluster")
plt.ylabel("Percentage of Applications")

plt.savefig(
    os.path.join(OUTPUT_DIR, "cluster_distribution.png"),
    dpi=300
)
plt.close()


# ============================================================
# NUMERICAL CLUSTER VISUALIZATIONS
# ============================================================

for col in numeric_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=cluster_df, x="kmeans_cluster", y=col, showfliers=False)

    plt.title("Cluster-wise Distribution: " + col)
    plt.xlabel("Cluster")
    plt.ylabel(col)

    plt.savefig(
        os.path.join(OUTPUT_DIR, "cluster_" + col + ".png"),
        dpi=300
    )
    plt.close()


# ============================================================
# CATEGORICAL CLUSTER VISUALIZATIONS
# ============================================================

for col in categorical_features:
    proportions = pd.crosstab(
        cluster_df["kmeans_cluster"],
        cluster_df[col],
        normalize="index"
    )
    
    proportions.plot(kind="bar", stacked=True, figsize=(12, 6), legend=False)

    plt.title("Cluster-wise Distribution: " + col)
    plt.xlabel("Cluster")
    plt.ylabel("Proportion")

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "categorical_" + col + ".png"
        ),
        dpi=300
    )
    plt.close()


# ============================================================
# CLUSTER-WISE MORTGAGE OUTCOME DISTRIBUTION
# ============================================================

outcome_df = pd.DataFrame({
    "kmeans_cluster": cluster_df["kmeans_cluster"],
    "outcome": df.loc[cluster_df.index, "action_taken_label"]
})

outcome_df = outcome_df[
    outcome_df["outcome"].isin([
        "Originated",
        "Approved Not Accepted",
        "Denied"
    ])
].copy()

outcome_distribution = pd.crosstab(
    outcome_df["kmeans_cluster"],
    outcome_df["outcome"],
    normalize="index"
) * 100

outcome_distribution = outcome_distribution.reindex(
    columns=[
        "Originated",
        "Approved Not Accepted",
        "Denied"
    ],
    fill_value=0
)

print("\nCluster-wise Mortgage Outcome Distribution:")
print(outcome_distribution.round(2))

outcome_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_outcome_distribution.csv"
    )
)

outcome_distribution.plot(kind="bar", stacked=True, figsize=(11, 6))

plt.title("Mortgage Outcomes Across K-Means Segments")
plt.xlabel("K-Means Cluster")
plt.ylabel("Percentage of Applications")
plt.legend(title="Mortgage Outcome")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cluster_outcome_distribution.png"
    ),
    dpi=300
)
plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CLUSTERING COMPLETED")
print("=" * 65)

print("K-Means Clusters:", optimal_k)
print("Hierarchical Clusters:", optimal_hierarchical_k)

print("\nK-Means Final Cluster Distribution:")
print(cluster_distribution)

print("\nResults saved at:", OUTPUT_DIR)
