import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import optuna

# For the reproducibility of results
RANDOM_STATE = 42
sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

# ==============================================================================
# MODEL REGISTRY
# ==============================================================================

MODELS = {
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "search_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        "search_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        "search_space": lambda trial: {
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "penalty": (
                "l2"
                if trial.params["solver"] == "lbfgs"
                else trial.suggest_categorical("penalty", ["l1", "l2"])
            ),
            "C": trial.suggest_float("C", 1e-4, 1.0, log=True),
        },
    },
}


# ==============================================================================
# VALIDATION PHASE (NESTED CV)
# ==============================================================================
def validate(model_name: str):
    """Performs nested cross-validation with Optuna-based hyperparameter optimization."""
    print(f"Running nested CV (Optuna) for model: {model_name}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data = pd.read_csv(PROJECT_ROOT / "data/train_set_cleaned.csv")

    X = data.drop(columns="player_won")
    y = data["player_won"]
    del data

    estimator = MODELS[model_name]["estimator"]
    search_space = MODELS[model_name]["search_space"]

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    outer_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n--- Outer Fold {fold_idx}/10 ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        def objective(trial):
            params = search_space(trial)
            model = estimator.set_params(**params)
            scores = cross_val_score(
                model, X_train, y_train, cv=inner_cv, scoring="accuracy"
            )
            return np.mean(scores)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        print(f"Best params (fold {fold_idx}): {study.best_params}")

        best_model = estimator.set_params(**study.best_params)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        outer_scores.append(acc)

        print(f"Fold {fold_idx} accuracy: {acc:.4f}")

    print("\n======================================")
    print(
        f"Nested CV Mean Accuracy: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}"
    )


# ==============================================================================
# TRAINING PHASE
# ==============================================================================
def train(model_name: str):
    """Runs Optuna on the full dataset and saves the best trained model."""
    print(f"Retraining final {model_name} model with Optuna...")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(PROJECT_ROOT / "data/train_set_cleaned.csv")
    X = data.drop(columns="player_won")
    y = data["player_won"]
    del data

    estimator = MODELS[model_name]["estimator"]
    search_space = MODELS[model_name]["search_space"]

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = search_space(trial)
        model = estimator.set_params(**params)
        score = cross_val_score(
            model, X, y, cv=inner_cv, scoring="balanced_accuracy", n_jobs=-1
        )
        return np.mean(score)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=300, show_progress_bar=True)

    print("\nBest parameters found by Optuna:", study.best_params)

    final_model = estimator.set_params(**study.best_params)
    final_model.fit(X, y)

    output_path = MODELS_DIR / f"{model_name}_optuna.pkl"
    joblib.dump(final_model, output_path)
    print(f"Final model saved to: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Model validation or training.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["validate", "train"],
        help="Choose 'validate' for nested CV or 'train' for Optuna retraining.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS.keys(),
        help=f"Model to run: {list(MODELS.keys())}",
    )

    args = parser.parse_args()

    if args.mode == "validate":
        validate(args.model)
    elif args.mode == "train":
        train(args.model)


if __name__ == "__main__":
    main()
