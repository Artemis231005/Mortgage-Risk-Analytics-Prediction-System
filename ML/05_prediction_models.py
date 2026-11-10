
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    make_scorer
)

from sklearn.inspection import permutation_importance
from sklearn.base import clone


RANDOM_STATE = 42
RESULTS_PATH = "Results/ML/Prediction_Results"
os.makedirs(RESULTS_PATH, exist_ok=True)


# Load Datasets
train_df = pd.read_csv("Results/Prepared_Datasets/Train.csv", low_memory=False)
val_df = pd.read_csv("Results/Prepared_Datasets/Validation.csv", low_memory=False)
test_df = pd.read_csv("Results/Prepared_Datasets/Test.csv", low_memory=False)

# print("Train Shape:", train_df.shape)
# print("Validation Shape:", val_df.shape)
# print("Test Shape:", test_df.shape)


# Define Features
numeric_features = [
    "loan_amount",
    "loan_to_value_ratio",
    "income",
    "property_value",
    "property_value_exempt",
    "tract_to_msa_income_percentage"
]

categorical_features = [
    col for col in train_df.columns
    if col not in numeric_features and col != "action_taken"
]

feature_columns = numeric_features + categorical_features

for df in [train_df, val_df, test_df]:
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_features:
        df[col] = df[col].fillna("Unknown").astype(str)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),

        "Balanced Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),

        "Macro Precision": precision_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),

        "Macro Recall": recall_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),

        "Macro F1": f1_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),

        "Weighted F1": f1_score(
            y_true, predictions,
            average="weighted",
            zero_division=0
        )
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def analyze_feature_importance(pipeline, X_val, y_val, model_name, model_type):
    model = pipeline.named_steps["model"]
    fitted_preprocessor = pipeline.named_steps["preprocessor"]

    transformed_features = fitted_preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Transformed_Feature": transformed_features,
        "Importance": model.feature_importances_
    })

    aggregated_importance = []

    for feature in feature_columns:
        if feature in numeric_features:
            matching = importance_df[
                importance_df["Transformed_Feature"] == "num__" + feature
            ]

        else:
            matching = importance_df[
                importance_df["Transformed_Feature"].str.startswith(
                    "cat__" + feature + "_"
                )
            ]

        aggregated_importance.append({
            "Feature": feature,
            "Built_In_Importance": matching["Importance"].sum()
        })

    built_in_df = pd.DataFrame(aggregated_importance)

    if model_type == "multiclass":
        scoring = make_scorer(
            f1_score,
            average="macro",
            zero_division=0
        )

    else:
        scoring = "average_precision"

    permutation_result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    permutation_df = pd.DataFrame({
        "Feature": X_val.columns,
        "Permutation_Importance": permutation_result.importances_mean,
        "Permutation_Std": permutation_result.importances_std
    })

    combined_df = built_in_df.merge(
        permutation_df,
        on="Feature"
    )

    combined_df = combined_df.sort_values(
        "Permutation_Importance",
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE:", model_name)
    print("=" * 65)

    print(combined_df.to_string(index=False))

    file_name = model_name.lower().replace(" ", "_")
    combined_df.to_csv(
        RESULTS_PATH + "/" + file_name + "_Feature_Importance.csv",
        index=False
    )
    
    return combined_df


# ============================================================
# MODEL 1: THREE-CLASS MORTGAGE OUTCOME PREDICTION
# ============================================================

print("\n\n")
print("=" * 65)
print("MODEL 1: THREE-CLASS MORTGAGE OUTCOME PREDICTION")
print("=" * 65)

# Define Target
X_train = train_df[feature_columns].copy()
y_train = train_df["action_taken"].astype(int)

X_val = val_df[feature_columns].copy()
y_val = val_df["action_taken"].astype(int)

X_test = test_df[feature_columns].copy()
y_test = test_df["action_taken"].astype(int)

classes = [1, 2, 3]

print("\nTraining Target Distribution:")
print(y_train.value_counts().sort_index())

print("\nTraining Target Percentages:")
print((y_train.value_counts(normalize=True).sort_index() * 100).round(2))


# TARGET ENCODING

class_mapping = {
    1: 0,
    2: 1,
    3: 2
}

reverse_mapping = {
    0: 1,
    1: 2,
    2: 3
}

y_train_encoded = y_train.map(class_mapping)
y_val_encoded = y_val.map(class_mapping)

class_weight_options = {
    "Moderate": {0: 1, 1: 2, 2: 1.5},
    "Higher Minority": {0: 1, 1: 3, 2: 2},
    "Strong Minority": {0: 1, 1: 5, 2: 2}
}


# ------------------------------------------------------------
# DEFINE MODELS
# ------------------------------------------------------------

models = {
    "Random Forest (Default)": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest (Moderate Weights)": RandomForestClassifier(
        class_weight=class_weight_options["Moderate"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest (Higher Minority Weights)": RandomForestClassifier(
        class_weight=class_weight_options["Higher Minority"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest (Strong Minority Weights)": RandomForestClassifier(
        class_weight=class_weight_options["Strong Minority"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

models["LightGBM (Default)"] = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    objective="multiclass",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

models["LightGBM (Balanced)"] = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    objective="multiclass",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

for weight_name, weights in class_weight_options.items():
    models["LightGBM (" + weight_name + " Weights)"] = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        objective="multiclass",
        class_weight=weights,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )


# ------------------------------------------------------------
# CLASSIFICATION THRESHOLD FUNCTION
# ------------------------------------------------------------

def apply_threshold(probabilities, threshold=None):
    if threshold is None:
        return np.argmax(probabilities, axis=1)

    predictions = []

    for row in probabilities:
        if row[1] >= threshold:
            predictions.append(1)

        elif row[0] >= row[2]:
            predictions.append(0)

        else:
            predictions.append(2)

    return np.array(predictions)


# ------------------------------------------------------------
# MODEL TRAINING AND THRESHOLD SELECTION
# ------------------------------------------------------------

results = []
trained_models = {}

thresholds = [None] + list(np.arange(0.05, 0.501, 0.025))

for name, model in models.items():
    print("\n" + "=" * 65)
    print("TRAINING:", name)
    print("=" * 65)

    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train_encoded)

    val_probabilities = pipeline.predict_proba(X_val)

    selected_threshold = None
    selected_predictions = None
    selected_metrics = None

    for threshold in thresholds:
        predictions = apply_threshold(val_probabilities, threshold)

        metrics = evaluate_model(y_val_encoded, predictions)

        if selected_metrics is None or metrics["Macro F1"] > selected_metrics["Macro F1"]:
            selected_threshold = threshold
            selected_predictions = predictions
            selected_metrics = metrics

    val_predictions = pd.Series(selected_predictions).map(reverse_mapping).values
    selected_metrics = evaluate_model(y_val, val_predictions)

    print("Selected Threshold:", selected_threshold)

    for metric_name, value in selected_metrics.items():
        print(metric_name + ":", round(value, 4))

    print("\nValidation Classification Report:")

    print(classification_report(
        y_val,
        val_predictions,
        labels=[1, 2, 3],
        target_names=[
            "Originated",
            "Approved Not Accepted",
            "Denied"
        ],
        zero_division=0
    ))

    result = {
        "Model": name,
        "Threshold": selected_threshold
    }
    result.update(selected_metrics)
    
    results.append(result)

    trained_models[name] = {
        "pipeline": pipeline,
        "threshold": selected_threshold
    }


# ------------------------------------------------------------
# MODEL COMPARISON SUMMARY
# ------------------------------------------------------------

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by=["Macro F1", "Balanced Accuracy"],
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 65)
print("MODEL 1 COMPARISON SUMMARY")
print("=" * 65)

print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    RESULTS_PATH + "/Model_Comparison_Optimized.csv",
    index=False
)


# ------------------------------------------------------------
# SELECT BEST MODEL
# ------------------------------------------------------------

best_model_name = comparison_df.iloc[0]["Model"]
model_1_threshold = comparison_df.iloc[0]["Threshold"]

if pd.isna(model_1_threshold):
    model_1_threshold = None

print("\nBest Model:", best_model_name)
print("Best Threshold:", model_1_threshold)

model_1_pipeline = clone(
    trained_models[best_model_name]["pipeline"]
)


# ------------------------------------------------------------
# MODEL 1 FEATURE IMPORTANCE
# ------------------------------------------------------------

analyze_feature_importance(
    pipeline=trained_models[best_model_name]["pipeline"],
    X_val=X_val,
    y_val=y_val_encoded,
    model_name="Model_1_Three_Class",
    model_type="multiclass"
)


# ------------------------------------------------------------
# FINAL TRAINING AND EVALUATION
# ------------------------------------------------------------

X_train_final = pd.concat([X_train, X_val], ignore_index=True)
y_train_final = pd.concat([y_train_encoded, y_val_encoded], ignore_index=True)

model_1_pipeline.fit(X_train_final, y_train_final)

test_probabilities = model_1_pipeline.predict_proba(X_test)
test_predictions_encoded = apply_threshold(test_probabilities, model_1_threshold)
test_predictions = pd.Series(test_predictions_encoded).map(reverse_mapping).values

test_metrics = evaluate_model(y_test,test_predictions)


print("\n" + "=" * 65)
print("MODEL 1 FINAL TEST RESULTS")
print("=" * 65)

for metric_name, value in test_metrics.items():
    print(metric_name + ":", round(value, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    test_predictions,
    labels=[1, 2, 3],
    target_names=[
        "Originated",
        "Approved Not Accepted",
        "Denied"
    ],
    zero_division=0
))


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[1, 2, 3]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 1", "Actual 2", "Actual 3"],
    columns=["Predicted 1", "Predicted 2", "Predicted 3"]
)

print("\nModel 1 Confusion Matrix:")
print(cm_df)

cm_df.to_csv(
    RESULTS_PATH + "/Confusion_Matrix_Optimized.csv"
)


# ============================================================
# MODEL 2: MORTGAGE APPROVAL CONVERSION PREDICTION
# ============================================================

print("\n\n")
print("=" * 65)
print("MODEL 2: MORTGAGE APPROVAL CONVERSION PREDICTION")
print("=" * 65)


# Prepare Binary Datasets
# 0 = Originated
# 1 = Approved Not Accepted

def prepare_conversion_data(df):
    conversion_df = df[df["action_taken"].isin([1, 2])].copy()
    
    X = conversion_df[feature_columns].copy()
    y = (conversion_df["action_taken"] == 2).astype(int)

    return X, y


X_train_binary, y_train_binary = prepare_conversion_data(train_df)
X_val_binary, y_val_binary = prepare_conversion_data(val_df)
X_test_binary, y_test_binary = prepare_conversion_data(test_df)

# print("\nBinary Dataset Shapes:")
# print("Train:", X_train_binary.shape)
# print("Validation:", X_val_binary.shape)
# print("Test:", X_test_binary.shape)

print("\nBinary Training Target Distribution:")
print(
    y_train_binary.value_counts().sort_index().rename({
        0: "Originated",
        1: "Approved Not Accepted"
    })
)

print("\nBinary Training Target Percentages:")
print(
    (y_train_binary.value_counts(normalize=True).sort_index() * 100)
    .round(2)
    .rename({
        0: "Originated",
        1: "Approved Not Accepted"
    })
)

binary_class_weight_options = {
    "Moderate": {0: 1, 1: 2},
    "Higher": {0: 1, 1: 3},
    "Strong": {0: 1, 1: 5}
}


# ------------------------------------------------------------
# DEFINE MODELS
# ------------------------------------------------------------

binary_models = {
    "Random Forest (Default)": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest (Moderate Weights)": RandomForestClassifier(
        class_weight=binary_class_weight_options["Moderate"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest (Higher Weights)": RandomForestClassifier(
        class_weight=binary_class_weight_options["Higher"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest (Strong Weights)": RandomForestClassifier(
        class_weight=binary_class_weight_options["Strong"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

binary_models["LightGBM (Default)"] = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    objective="binary",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

binary_models["LightGBM (Balanced)"] = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    objective="binary",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

for weight_name, weights in binary_class_weight_options.items():
    binary_models["LightGBM (" + weight_name + " Weights)"] = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        objective="binary",
        class_weight=weights,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )


# ------------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_binary_model(y_true, predictions, probabilities):
    metrics = evaluate_model(y_true, predictions)
    metrics["PR-AUC"] = average_precision_score(y_true, probabilities)
    
    return metrics


# ------------------------------------------------------------
# MODEL TRAINING AND THRESHOLD SELECTION
# ------------------------------------------------------------

binary_thresholds = np.arange(0.05, 0.951, 0.025)

binary_results = []
binary_trained_models = {}

for name, model in binary_models.items():
    print("\n" + "=" * 65)
    print("TRAINING:", name)
    print("=" * 65)

    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", model)
    ])

    pipeline.fit(X_train_binary, y_train_binary)

    val_probabilities = pipeline.predict_proba(X_val_binary)[:, 1]

    selected_threshold = None
    selected_predictions = None
    selected_metrics = None

    for threshold in binary_thresholds:
        predictions = (val_probabilities >= threshold).astype(int)

        metrics = evaluate_binary_model(y_val_binary, predictions, val_probabilities)

        if selected_metrics is None or metrics["Macro F1"] > selected_metrics["Macro F1"]:
            selected_threshold = threshold
            selected_predictions = predictions
            selected_metrics = metrics

    print("Selected Threshold:", round(selected_threshold, 3))

    for metric_name, value in selected_metrics.items():
        print(metric_name + ":", round(value, 4))

    print("\nValidation Classification Report:")

    print(classification_report(
        y_val_binary,
        selected_predictions,
        labels=[0, 1],
        target_names=[
            "Originated",
            "Approved Not Accepted"
        ],
        zero_division=0
    ))

    result = {
        "Model": name,
        "Threshold": selected_threshold
    }

    result.update(selected_metrics)
    binary_results.append(result)

    binary_trained_models[name] = {
        "pipeline": pipeline,
        "threshold": selected_threshold
    }


# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------

binary_comparison_df = pd.DataFrame(binary_results)

binary_comparison_df = binary_comparison_df.sort_values(
    by=["Macro F1", "PR-AUC", "Balanced Accuracy"],
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 65)
print("MODEL 2 COMPARISON")
print("=" * 65)

print(binary_comparison_df.to_string(index=False))

binary_comparison_df.to_csv(
    RESULTS_PATH + "/Model_2_Conversion_Comparison.csv",
    index=False
)

# ------------------------------------------------------------
# SELECT BEST MODEL
# ------------------------------------------------------------

best_binary_model_name = binary_comparison_df.iloc[0]["Model"]
model_2_threshold = binary_comparison_df.iloc[0]["Threshold"]

print("\nBest Binary Model:", best_binary_model_name)
print("Best Binary Threshold:", round(model_2_threshold, 3))

model_2_pipeline = clone(
    binary_trained_models[best_binary_model_name]["pipeline"]
)


# ------------------------------------------------------------
# MODEL 2 FEATURE IMPORTANCE
# ------------------------------------------------------------

analyze_feature_importance(
    pipeline=binary_trained_models[best_binary_model_name]["pipeline"],
    X_val=X_val_binary,
    y_val=y_val_binary,
    model_name="Model_2_Conversion",
    model_type="binary"
)


# ------------------------------------------------------------
# MODEL 2: THRESHOLD TRADE-OFF ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MODEL 2: THRESHOLD TRADE-OFF ANALYSIS")
print("=" * 65)

selected_binary_pipeline = binary_trained_models[best_binary_model_name]["pipeline"]
threshold_probabilities = selected_binary_pipeline.predict_proba(X_val_binary)[:, 1]

threshold_results = []

for threshold in binary_thresholds:
    predictions = (threshold_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val_binary,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val_binary,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val_binary,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val_binary,
        predictions,
        zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    threshold_results.append({
        "Threshold": round(threshold, 3),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "True_Positive": tp,
        "False_Positive": fp,
        "True_Negative": tn,
        "False_Negative": fn,
        "Flagged_Applications": tp + fp
    })


threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold Comparison:")
print(threshold_df.round(4).to_string(index=False))

threshold_df.to_csv(
    RESULTS_PATH + "/Model_2_Threshold_Tradeoff.csv",
    index=False
)

selected_row = threshold_df.iloc[
    (threshold_df["Threshold"] - model_2_threshold).abs().argmin()
]

print("\nCurrently Selected Threshold:", model_2_threshold)

print("\nSelected Threshold Metrics:")
print(selected_row.to_string())


# TOP THRESHOLDS

print("\nTop Thresholds by Recall:")
print(
    threshold_df.sort_values(
        by=["Recall", "Precision"],
        ascending=False
    ).head(5).round(4).to_string(index=False)
)

print("\nTop Thresholds by Precision:")
print(
    threshold_df.sort_values(
        by=["Precision", "Recall"],
        ascending=False
    ).head(5).round(4).to_string(index=False)
)


# ------------------------------------------------------------
# FINAL TRAINING AND EVALUATION
# ------------------------------------------------------------

X_binary_train_final = pd.concat([X_train_binary, X_val_binary], ignore_index=True)
y_binary_train_final = pd.concat([y_train_binary, y_val_binary], ignore_index=True)

model_2_pipeline.fit(X_binary_train_final, y_binary_train_final)


binary_test_probabilities = model_2_pipeline.predict_proba(X_test_binary)[:, 1]
binary_test_predictions = (binary_test_probabilities >= model_2_threshold).astype(int)

binary_test_metrics = evaluate_binary_model(y_test_binary, binary_test_predictions, binary_test_probabilities)


print("\n" + "=" * 65)
print("MODEL 2 FINAL TEST RESULTS")
print("=" * 65)

for metric_name, value in binary_test_metrics.items():
    print(metric_name + ":", round(value, 4))

print("\nClassification Report:")

print(classification_report(
    y_test_binary,
    binary_test_predictions,
    labels=[0, 1],
    target_names=[
        "Originated",
        "Approved Not Accepted"
    ],
    zero_division=0
))


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

binary_cm = confusion_matrix(
    y_test_binary,
    binary_test_predictions,
    labels=[0, 1]
)

binary_cm_df = pd.DataFrame(
    binary_cm,
    index=["Actual Originated", "Actual Approved Not Accepted"],
    columns=["Predicted Originated", "Predicted Approved Not Accepted"]
)

print("\nModel 2 Confusion Matrix:")
print(binary_cm_df)

binary_cm_df.to_csv(
    RESULTS_PATH + "/Model_2_Conversion_Confusion_Matrix.csv"
)


# ------------------------------------------------------------
# SAVE BINARY TEST PREDICTIONS
# ------------------------------------------------------------

binary_prediction_df = pd.DataFrame({
    "Actual": y_test_binary.values,
    "Predicted": binary_test_predictions,
    "Probability_Originated": 1 - binary_test_probabilities,
    "Probability_Approved_Not_Accepted": binary_test_probabilities
})

binary_prediction_df.to_csv(
    RESULTS_PATH + "/Model_2_Conversion_Test_Predictions.csv",
    index=False
)


# ============================================================
# SAVE BOTH MODELS
# ============================================================

model_bundle = {
    "model_1_three_class": {
        "pipeline": model_1_pipeline,
        "class_mapping": class_mapping,
        "reverse_mapping": reverse_mapping,
        "target_classes": classes,
        "best_model_name": best_model_name,
        "best_threshold": model_1_threshold,
        "class_weight_options": class_weight_options,
        "feature_columns": feature_columns
    },

    "model_2_conversion": {
        "pipeline": model_2_pipeline,
        "target_mapping": {
            0: "Originated",
            1: "Approved Not Accepted"
        },
        "best_model_name": best_binary_model_name,
        "best_threshold": model_2_threshold,
        "class_weight_options": binary_class_weight_options,
        "feature_columns": feature_columns
    }
}

joblib.dump(
    model_bundle,
    RESULTS_PATH + "/Best_Models.pkl"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("BOTH MORTGAGE PREDICTION MODELS COMPLETED")
print("=" * 75)

print("\nMODEL 1:")
print("Best Model:", best_model_name)
print("Threshold:", model_1_threshold)
print("Test Macro F1:", round(test_metrics["Macro F1"], 4))

print("\nMODEL 2:")
print("Best Model:", best_binary_model_name)
print("Threshold:", round(model_2_threshold, 3))
print("Test Macro F1:", round(binary_test_metrics["Macro F1"], 4))
print("Test PR-AUC:", round(binary_test_metrics["PR-AUC"], 4))

print("\nFiles Saved at: ")
print(RESULTS_PATH)
