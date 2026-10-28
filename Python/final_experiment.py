# ============================================================
# FINAL RETAIL SALES PREDICTION EXPERIMENT
# Publication-ready experimental pipeline
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(PROJECT_PATH, "Dataset")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "Output")
GRAPH_PATH = os.path.join(PROJECT_PATH, "Graphs")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

orders2015 = pd.read_csv(f"{DATASET_PATH}/order_2015.csv")
orders2016 = pd.read_csv(f"{DATASET_PATH}/order_2016.csv")
orders2017 = pd.read_csv(f"{DATASET_PATH}/order_2017.csv")
orders2018 = pd.read_csv(f"{DATASET_PATH}/order_2018.csv")

people = pd.read_csv(f"{DATASET_PATH}/People.csv")
returns = pd.read_csv(f"{DATASET_PATH}/Returns.csv")
target = pd.read_csv(f"{DATASET_PATH}/Target.csv")

orders = pd.concat(
    [orders2015, orders2016, orders2017, orders2018],
    ignore_index=True
)

print("Raw order rows:", len(orders))


# ============================================================
# 3. REMOVE EXACT DUPLICATES
# ============================================================

duplicate_count = orders.duplicated().sum()

print("Duplicate rows:", duplicate_count)

orders = orders.drop_duplicates().copy()

print("Rows after duplicate removal:", len(orders))


# ============================================================
# 4. DATE CONVERSION
# ============================================================

orders["Order Date"] = pd.to_datetime(
    orders["Order Date"],
    unit="D",
    origin="1899-12-30"
)

orders["Ship Date"] = pd.to_datetime(
    orders["Ship Date"],
    unit="D",
    origin="1899-12-30"
)


# ============================================================
# 5. TEMPORAL FEATURES
# ============================================================

orders["Year"] = orders["Order Date"].dt.year
orders["Month"] = orders["Order Date"].dt.month
orders["Quarter"] = orders["Order Date"].dt.quarter


# ============================================================
# 6. PEOPLE MERGE
# ============================================================

print("\nPeople table rows:", len(people))
print("People duplicate regions:", people["Region"].duplicated().sum())

rows_before_people = len(orders)

orders = pd.merge(
    orders,
    people,
    on="Region",
    how="left",
    validate="many_to_one"
)

rows_after_people = len(orders)

print("Rows before People merge:", rows_before_people)
print("Rows after People merge:", rows_after_people)


# ============================================================
# 7. RETURNS CLEANING
# ============================================================

print("\nDuplicate Order IDs in Returns:",
      returns["Order ID"].duplicated().sum())

returns = returns.drop_duplicates(
    subset="Order ID"
).copy()

orders = pd.merge(
    orders,
    returns,
    on="Order ID",
    how="left",
    validate="many_to_one"
)

orders["Returned"] = orders["Returned"].fillna("No")


# ============================================================
# 8. SALES TARGET CLEANING
# ============================================================

# Convert target values such as 500K to 500000.
target["Sales Target"] = (
    target["Sales Target"]
    .astype(str)
    .str.replace("K", "", regex=False)
    .astype(float)
    * 1000
)


# ============================================================
# 9. SALES TARGET MERGE
# ============================================================

orders = pd.merge(
    orders,
    target,
    on=["Category", "Year"],
    how="left",
    validate="many_to_one"
)

# ============================================================
# DATA INTEGRITY DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("DATA INTEGRITY DIAGNOSTICS")
print("=" * 70)

print("\nMissing values:")
print(orders.isnull().sum())

print("\nMissing Quantity:")
print(orders["Quantity"].isnull().sum())

print("\nDuplicate Regions in People:")
print(people["Region"].duplicated().sum())

print("\nFinal number of rows:")
print(len(orders))

print("\nFinal number of columns:")
print(len(orders.columns))

# ============================================================
# 10. QUANTITY CLEANING
# ============================================================

quantity_mapping = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10
}

orders["Quantity"] = orders["Quantity"].replace(quantity_mapping)

orders["Quantity"] = pd.to_numeric(
    orders["Quantity"],
    errors="coerce"
)


# IMPORTANT:
# Do not silently replace missing Quantity with zero unless
# your source data analysis establishes that zero is appropriate.
#
# For now, remove records with missing modeling variables later.


# ============================================================
# 11. FINAL DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET VALIDATION")
print("=" * 70)

print("Shape:", orders.shape)

print("\nMissing values:")
print(orders.isnull().sum())

print("\nDuplicate rows:", orders.duplicated().sum())

print("\nYear distribution:")
print(orders["Year"].value_counts().sort_index())


# Save integrated dataset
orders.to_csv(
    f"{OUTPUT_PATH}/final_validated_dataset.csv",
    index=False
)


# ============================================================
# 12. MODELING DATA
# ============================================================

features = [
    "Category",
    "Sub-Category",
    "Region",
    "Segment",
    "Quantity",
    "Discount",
    "Year",
    "Month",
    "Quarter"
]
# Primary feature set excludes Sales Target
# Sales Target is evaluated separately as a sensitivity feature.

features_with_target = features + ["Sales Target"]

target_variable = "Sales"

model_data = orders[
    features_with_target + [target_variable]
].dropna().copy()

# Primary experiment: WITHOUT Sales Target
X = model_data[features]
y = model_data[target_variable]


# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 14. PREPROCESSING
# ============================================================

categorical_features = [
    "Category",
    "Sub-Category",
    "Region",
    "Segment"
]

numerical_features = [
    "Quantity",
    "Discount",
    "Year",
    "Month",
    "Quarter"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 15. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=3,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1
    )
}


# ============================================================
# 16. PRIMARY TEST-SET EXPERIMENT
# ============================================================

results = []
predictions_dict = {}

print("\n" + "=" * 70)
print("PRIMARY TEST-SET EXPERIMENT")
print("=" * 70)

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    predictions_dict[name] = predictions

    print(
        f"{name:20s} | "
        f"MAE={mae:.3f} | "
        f"RMSE={rmse:.3f} | "
        f"R2={r2:.4f}"
    )


results_df = pd.DataFrame(results)

print("\nTABLE IV")
print(results_df)

results_df.to_csv(
    f"{OUTPUT_PATH}/table_IV_test_results.csv",
    index=False
)

# ============================================================
# FIGURE 1 — TEST-SET R2
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.xlabel("Model")
plt.ylabel("R²")
plt.title("Test-Set R² Across Regression Models")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_PATH}/fig1_test_set_r2.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 1 saved:")
print(f"{GRAPH_PATH}/fig1_test_set_r2.png")



# ============================================================
# 17. FIVE-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FIVE-FOLD CROSS VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name, model in models.items():

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for train_idx, val_idx in kf.split(X):

        X_cv_train = X.iloc[train_idx]
        X_cv_val = X.iloc[val_idx]

        y_cv_train = y.iloc[train_idx]
        y_cv_val = y.iloc[val_idx]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_cv_train, y_cv_train)

        pred = pipeline.predict(X_cv_val)

        fold_mae.append(
            mean_absolute_error(y_cv_val, pred)
        )

        fold_rmse.append(
            np.sqrt(
                mean_squared_error(y_cv_val, pred)
            )
        )

        fold_r2.append(
            r2_score(y_cv_val, pred)
        )

    cv_results.append({
        "Model": name,
        "MAE Mean": np.mean(fold_mae),
        "MAE SD": np.std(fold_mae, ddof=1),
        "RMSE Mean": np.mean(fold_rmse),
        "RMSE SD": np.std(fold_rmse, ddof=1),
        "R2 Mean": np.mean(fold_r2),
        "R2 SD": np.std(fold_r2, ddof=1)
    })


cv_df = pd.DataFrame(cv_results)

print("\nTABLE V")
print(cv_df)

cv_df.to_csv(
    f"{OUTPUT_PATH}/table_V_cross_validation.csv",
    index=False
)

# ============================================================
# 18. FEATURE-GROUP ABLATION
# ============================================================

CATEGORICAL_ALL = [
    "Category",
    "Sub-Category",
    "Region",
    "Segment"
]

NUMERICAL_ALL = [
    "Quantity",
    "Discount",
    "Year",
    "Month",
    "Quarter",
    "Sales Target"
]


def evaluate_feature_set(
    feature_list,
    experiment_name
):

    X_ablation = model_data[feature_list]
    y_ablation = model_data[target_variable]

    Xtr, Xte, ytr, yte = train_test_split(
        X_ablation,
        y_ablation,
        test_size=0.20,
        random_state=42
    )

    # Explicitly identify categorical and numerical
    # variables instead of relying on pandas dtype.
    categorical = [
        col for col in feature_list
        if col in CATEGORICAL_ALL
    ]

    numerical = [
        col for col in feature_list
        if col in NUMERICAL_ALL
    ]

    processor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                categorical
            ),
            (
                "numerical",
                "passthrough",
                numerical
            )
        ],
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", processor),
            ("model", LinearRegression())
        ]
    )

    pipeline.fit(Xtr, ytr)

    pred = pipeline.predict(Xte)

    return {
        "Experiment": experiment_name,
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": np.sqrt(
            mean_squared_error(yte, pred)
        ),
        "R2": r2_score(yte, pred)
    }
# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

ablation_results = []

# Experiment 1: Quantity + Discount
ablation_results.append(
    evaluate_feature_set(
        ["Quantity", "Discount"],
        "Basic"
    )
)

# Experiment 2: Quantity + Discount + temporal features
ablation_results.append(
    evaluate_feature_set(
        [
            "Quantity",
            "Discount",
            "Year",
            "Month",
            "Quarter"
        ],
        "Basic + temporal"
    )
)

# Experiment 3: Full primary feature set WITHOUT Sales Target
ablation_results.append(
    evaluate_feature_set(
        features,
        "Full without Sales Target"
    )
)

# Experiment 4: Full primary feature set WITH Sales Target
ablation_results.append(
    evaluate_feature_set(
        features_with_target,
        "Full + Sales Target"
    )
)

ablation_df = pd.DataFrame(ablation_results)

print("\nTABLE VI")
print(ablation_df)

ablation_df.to_csv(
    f"{OUTPUT_PATH}/table_VI_ablation.csv",
    index=False
)

# ============================================================
# FIGURE 3 — FEATURE-GROUP ABLATION R2
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    ablation_df["Experiment"],
    ablation_df["R2"]
)

plt.xlabel("Feature Set")
plt.ylabel("R²")
plt.title("R² Across Feature-Group Ablation Experiments")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_PATH}/fig3_feature_ablation_r2.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 3 saved:")
print(f"{GRAPH_PATH}/fig3_feature_ablation_r2.png")

# ============================================================
# 19. SALES TARGET SENSITIVITY
# ============================================================

print("\n" + "=" * 70)
print("SALES TARGET SENSITIVITY")
print("=" * 70)

sensitivity_data = orders[
    features_with_target + [target_variable]
].dropna().copy()

X_target = sensitivity_data[features_with_target]
y_target = sensitivity_data[target_variable]

Xtr, Xte, ytr, yte = train_test_split(
    X_target,
    y_target,
    test_size=0.20,
    random_state=42
)

# Explicit feature types for the Sales Target sensitivity experiment

cat_target = [
    "Category",
    "Sub-Category",
    "Region",
    "Segment"
]

num_target = [
    "Quantity",
    "Discount",
    "Year",
    "Month",
    "Quarter",
    "Sales Target"
]

target_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            cat_target
        ),
        (
            "numerical",
            "passthrough",
            num_target
        )
    ]
)

target_results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", target_preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(Xtr, ytr)

    pred = pipeline.predict(Xte)

    target_results.append({
        "Model": name,
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "R2": r2_score(yte, pred)
    })


target_df = pd.DataFrame(target_results)

print("\nTABLE VII")
print(target_df)

target_df.to_csv(
    f"{OUTPUT_PATH}/table_VII_sales_target.csv",
    index=False
)


# ============================================================
# 20. 2018 TEMPORAL HOLDOUT
# ============================================================

print("\n" + "=" * 70)
print("2018 TEMPORAL HOLDOUT")
print("=" * 70)

temporal_features = features

temporal_data = orders[
    temporal_features + [target_variable]
].dropna().copy()

train_temporal = temporal_data[
    temporal_data["Year"] < 2018
]

test_temporal = temporal_data[
    temporal_data["Year"] == 2018
]

X_temporal_train = train_temporal[temporal_features]
y_temporal_train = train_temporal[target_variable]

X_temporal_test = test_temporal[temporal_features]
y_temporal_test = test_temporal[target_variable]

temporal_results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_temporal_train,
        y_temporal_train
    )

    pred = pipeline.predict(
        X_temporal_test
    )

    temporal_results.append({
        "Model": name,
        "MAE": mean_absolute_error(
            y_temporal_test,
            pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_temporal_test,
                pred
            )
        ),
        "R2": r2_score(
            y_temporal_test,
            pred
        )
    })


temporal_results_df = pd.DataFrame(
    temporal_results
)

print("\nTABLE VIII")
print(temporal_results_df)

temporal_results_df.to_csv(
    f"{OUTPUT_PATH}/table_VIII_temporal.csv",
    index=False
)

# ============================================================
# FIGURE 4 — 2018 TEMPORAL HOLDOUT R2
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    temporal_results_df["Model"],
    temporal_results_df["R2"]
)

plt.xlabel("Model")
plt.ylabel("R²")
plt.title("2018 Temporal Holdout R²")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_PATH}/fig4_2018_temporal_holdout_r2.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 4 saved:")
print(
    f"{GRAPH_PATH}/fig4_2018_temporal_holdout_r2.png"
)

# ============================================================
# 21. ERROR ANALYSIS — GRADIENT BOOSTING
# ============================================================

gb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=3,
                random_state=42
            )
        )
    ]
)

gb_pipeline.fit(X_train, y_train)

gb_predictions = gb_pipeline.predict(X_test)

error_df = X_test.copy()

error_df["Actual Sales"] = y_test.values
error_df["Predicted Sales"] = gb_predictions
error_df["Error"] = (
    error_df["Predicted Sales"]
    - error_df["Actual Sales"]
)

error_df["Absolute Error"] = (
    error_df["Error"].abs()
)

error_df["Negative Prediction"] = (
    error_df["Predicted Sales"] < 0
)


# ============================================================
# 22. SALES QUARTILES
# ============================================================

error_df["Sales Quartile"] = pd.qcut(
    error_df["Actual Sales"],
    q=4,
    labels=[
        "Q1 lowest",
        "Q2",
        "Q3",
        "Q4 highest"
    ]
)

quartile_results = []

for quartile, group in error_df.groupby(
    "Sales Quartile",
    observed=True
):

    quartile_results.append({
        "Sales Quartile": quartile,
        "N": len(group),
        "MAE": mean_absolute_error(
            group["Actual Sales"],
            group["Predicted Sales"]
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                group["Actual Sales"],
                group["Predicted Sales"]
            )
        ),
        "Mean Error": group["Error"].mean(),
        "Neg. Pred. %": (
            group["Negative Prediction"].mean()
            * 100
        )
    })


quartile_df = pd.DataFrame(
    quartile_results
)

print("\nTABLE IX")
print(quartile_df)

quartile_df.to_csv(
    f"{OUTPUT_PATH}/table_IX_sales_quartile.csv",
    index=False
)


# ============================================================
# 23. CATEGORY ERROR
# ============================================================

category_results = []

for category, group in error_df.groupby(
    "Category"
):

    category_results.append({
        "Category": category,
        "N": len(group),
        "MAE": mean_absolute_error(
            group["Actual Sales"],
            group["Predicted Sales"]
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                group["Actual Sales"],
                group["Predicted Sales"]
            )
        ),
        "Neg. Pred. %": (
            group["Negative Prediction"].mean()
            * 100
        )
    })


category_df = pd.DataFrame(
    category_results
)

print("\nTABLE X")
print(category_df)

category_df.to_csv(
    f"{OUTPUT_PATH}/table_X_category.csv",
    index=False
)


# ============================================================
# 24. REGION ERROR
# ============================================================

region_results = []

for region, group in error_df.groupby(
    "Region"
):

    region_results.append({
        "Region": region,
        "N": len(group),
        "MAE": mean_absolute_error(
            group["Actual Sales"],
            group["Predicted Sales"]
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                group["Actual Sales"],
                group["Predicted Sales"]
            )
        ),
        "Neg. Pred. %": (
            group["Negative Prediction"].mean()
            * 100
        )
    })


region_df = pd.DataFrame(
    region_results
)

print("\nTABLE XI")
print(region_df)

region_df.to_csv(
    f"{OUTPUT_PATH}/table_XI_region.csv",
    index=False
)


# ============================================================
# 25. FIGURE 5 — ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    gb_predictions,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(
    "Actual versus Predicted Sales — Gradient Boosting"
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_PATH}/fig5_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 26. FIGURE 6 — MAE BY SALES QUARTILE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    quartile_df["Sales Quartile"],
    quartile_df["MAE"]
)

plt.xlabel("Sales Quartile")
plt.ylabel("MAE")

plt.title(
    "Gradient Boosting MAE by Actual-Sales Quartile"
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_PATH}/fig6_mae_by_quartile.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

# Transform data using the fitted preprocessing step.
#
# Gradient Boosting operates on the transformed feature matrix.
# We aggregate one-hot encoded feature importances back to
# their original feature groups.

fitted_preprocessor = gb_pipeline.named_steps[
    "preprocessor"
]

gb_model = gb_pipeline.named_steps[
    "model"
]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importances = gb_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})


def original_feature_name(name):

    if "__" in name:
        name = name.split("__", 1)[1]

    for prefix in [
        "Category_",
        "Sub-Category_",
        "Region_",
        "Segment_"
    ]:

        if name.startswith(prefix):
            return prefix.rstrip("_")

    if name in [
        "Quantity",
        "Discount",
        "Year",
        "Month",
        "Quarter"
    ]:
        return name

    return name


importance_df["Feature Group"] = (
    importance_df["Feature"]
    .apply(original_feature_name)
)

grouped_importance = (
    importance_df
    .groupby("Feature Group")["Importance"]
    .sum()
    .sort_values(ascending=False)
)


print("\nFEATURE IMPORTANCE")
print(grouped_importance)


grouped_importance.to_csv(
    f"{OUTPUT_PATH}/feature_importance.csv"
)


# ============================================================
# 28. FIGURE 7
# ============================================================

plt.figure(figsize=(8, 5))

grouped_importance.sort_values().plot(
    kind="barh"
)

plt.xlabel("Aggregated Feature Importance")
plt.ylabel("Feature Group")

plt.title(
    "Aggregated Gradient Boosting Feature Importance"
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_PATH}/fig7_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

print("\nOutput files saved in:")
print(OUTPUT_PATH)

print("\nGraph files saved in:")
print(GRAPH_PATH)

print("\nIMPORTANT:")
print("Compare the regenerated results with the current paper")
print("before changing any manuscript tables.")