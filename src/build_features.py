import pandas as pd
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from feature_transformers import (
    HPFeaturesExtractor,
    StatusDiffExtractor,
    DropColumnTransformer,
    MoveFeatureExtractor,
    MissDiffExtractor,
    SwitchFeatureExtractor,
    BattleFeatureExtractor,
)

# ==============================================================================
# 1. CONFIGURATION AND PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input paths
train_set_path = PROJECT_ROOT / "data" / "train.jsonl"
test_set_path = PROJECT_ROOT / "data" / "test.jsonl"

# Output paths
train_output_path = PROJECT_ROOT / "data" / "train_set_cleaned.csv"
test_output_path = PROJECT_ROOT / "data" / "test_set_cleaned.csv"

# ==============================================================================
# 2. DATA LOADING AND PREPARATION
# ==============================================================================
print("Loading data...")

# Load training data
X_train = pd.read_json(train_set_path, lines=True)
# Remove known problematic row, its label is wrong
X_train.drop(4877, inplace=True)
y = X_train.pop("player_won")
# Load test data
X_test = pd.read_json(test_set_path, lines=True)

print("Data loaded successfully.")

# ==============================================================================
# 3. PIPELINE DEFINITION
# ==============================================================================
set_config(transform_output="pandas")
# Define which columns to drop after feature extraction
columns_to_drop = [
    "battle_id",
    "p1_team_details",
    "p2_lead_details",
    "battle_timeline",
]

# Create the full data processing pipeline
data_processing_pipeline = Pipeline(
    [
        ("hp_features", HPFeaturesExtractor()),
        ("status_features", StatusDiffExtractor()),
        ("miss_features", MissDiffExtractor()),
        ("moves_features", MoveFeatureExtractor()),
        ("battle_features", BattleFeatureExtractor()),
        ("drop_columns", DropColumnTransformer(columns_to_drop=columns_to_drop)),
        ("scaler", StandardScaler()),
    ]
)

# ==============================================================================
# 4. PIPELINE EXECUTION
# ==============================================================================
print("Processing the training set...")
X_train_clean = data_processing_pipeline.fit_transform(X_train)

print("Processing the test set...")
X_test_clean = data_processing_pipeline.fit_transform(X_test)

# ==============================================================================
# 5. SAVE PROCESSED DATA
# ==============================================================================
X_train_clean["player_won"] = y
X_train_clean.to_csv(train_output_path, index=False)
X_test_clean.to_csv(test_output_path, index=False)

print(f"Processed files saved successfully!")
print(f" - Training data: {train_output_path}")
print(f" - Test data:     {test_output_path}")

print("\nColumns in the cleaned training DataFrame:")
print(X_train_clean.columns.tolist())
