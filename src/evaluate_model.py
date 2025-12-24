import argparse
import joblib
import pandas as pd
from pathlib import Path
import sys


# ==============================================================================
# 1. CORE PREDICTION LOGIC
# ==============================================================================
def run_predictions(model_name: str):
    """
    Loads a trained model and a test set, generates predictions,
    and saves them to a submission file.
    """

    # --- 1.1 PATH DEFINITIONS ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

    # Create a path for the model-specific subfolder
    MODEL_SUBMISSION_DIR = SUBMISSIONS_DIR / model_name
    MODEL_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # The final output path now points inside the new subfolder
    SUBMISSION_PATH = MODEL_SUBMISSION_DIR / "submission.csv"

    # --- 1.2 MODEL AND TEST DATA LOADING ---
    print(f"Phase 1: Loading the '{model_name}' model...")
    model_path = MODELS_DIR / f"{model_name}_optuna.pkl"

    try:
        trained_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{model_path}'.", file=sys.stderr)
        print(
            f"Please run 'python src/train_model.py --model {model_name}' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred while loading the model: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading the test set...")
    test_data_path = DATA_DIR / "test_set_cleaned.csv"

    try:
        X_test = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(
            f"ERROR: Test data file not found at '{test_data_path}'.", file=sys.stderr
        )
        sys.exit(1)

    print("Data loaded successfully.")

    # --- 1.3 PREDICTION GENERATION AND SUBMISSION FILE CREATION ---
    print(f"Phase 2: Generating predictions...")

    try:
        y_pred_test = trained_model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        print(
            "This may be due to a mismatch in features (columns) between training and test data.",
            file=sys.stderr,
        )
        sys.exit(1)

    if "battle_id" in X_test.columns:
        battle_ids = X_test["battle_id"]
    else:
        battle_ids = range(len(X_test))

    submission_df = pd.DataFrame(
        {"battle_id": battle_ids, "player_won": y_pred_test.astype(int)}
    )

    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nSubmission file created successfully!")
    print(f"Saved to: {SUBMISSION_PATH}")
    print("\nFile preview:")
    print(submission_df.head())


# ==============================================================================
# 2. SCRIPT EXECUTION
# ==============================================================================
def main():
    """Parses arguments and runs the prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using a trained model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to use for predictions (e.g., random_forest).",
    )
    args = parser.parse_args()

    run_predictions(model_name=args.model)


if __name__ == "__main__":
    main()
