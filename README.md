## Pokémon Battle Outcome Prediction

This project implements a **machine learning pipeline** to predict the winner of a Pokémon battle by analyzing a timeline of up to **30 turns of in-game events**.

The project is based on the **FDS: Pokémon Battles Prediction 2025** Kaggle competition, proposed by **Indro Spinelli, Leonardo Rocci, and Simone Facchiano** as part of the *Foundations of Data Science* course at **Sapienza University of Rome (2025)**.

🔗 **Kaggle Competition:**  
https://kaggle.com/competitions/fds-pokemon-battles-prediction-2025


## Project Structure

The repository is organized as follows to clearly separate the different components:

- **`/data`**: Contains the datasets, both raw (`.jsonl`) and processed (`_cleaned.csv`).
- **`/models`**: Saves the trained model objects (`.pkl` files).
- **`/notebooks`**: Contains Jupyter Notebooks used for exploratory data analysis (EDA).
- **`/submissions`**: Saves the final `submission.csv` files, organized into subfolders for each model.
- **`/src`**: Contains all the project's source code.
    - `dataset.py`: Script to download and extract the initial dataset.
    - `build_features.py`: Runs the preprocessing pipeline to clean data and create new features.
    - `feature_transformers.py`: Defines custom `scikit-learn` transformer classes.
    - `train_model.py`: Trains a chosen model using `Optuna` to find the best hyperparameters.
    - `evaluate_model.py`: Loads a trained model to generate predictions on the test set.

---

## Setup and Installation

To run the project, follow these steps to set up the development environment.

**1. Clone the repository and set the environment**
```bash
git clone https://github.com/Catairu/Pokemon-Battle-Predictor.git
cd Pokemon-Battle-Predictor
uv sync
```

---

## Execution Workflow 

Run the scripts in this order to reproduce the entire process, from data download to submission file generation.

### **Step 1: Download the Dataset**

This command downloads the .zip file from Google Drive, extracts it into the /data folder, and finally removes the archive to keep the directory clean.

```bash
uv run python src/dataset.py
```

### **Step 2: Data Preparation and Cleaning**

This script applies the feature engineering pipeline to the `train.jsonl` and `test.jsonl` files. The resulting files (`train_set_cleaned.csv` and `test_set_cleaned.csv`) will be saved in the `/data` folder.

```bash
uv run python src/build_features.py
```

### **Step 3: Model Training**

Use this script to train one of the available models. The best model, identified via `Optuna`, will be saved as a `.pkl` file in the `/models` folder.

Choose which model to train using the `--model` argument.

You also have to use the `--mode` argument to specify the script's behavior: use `train` to run the full Optuna and save the best model, or `validate` to perform `NestedCV` for a performance check.


```bash
uv run python src/train_model.py --mode train --model random_forest

uv run python src/train_model.py --mode validate --model xgboost
```

**Available Models**: `random_forest`, `xgboost`, `logistic_regression`.

### **Step 4: Generation of Predictions**

Once a model has been trained, use this script to load the corresponding `.pkl` file and generate the submission file.

The result will be saved in a dedicated subfolder within `/submissions`.

```bash

uv run python src/evaluate_model.py --model random_forest

uv run python src/evaluate_model.py --model xgboost
```
The output of the first command, for example, will be saved in `submissions/random_forest/submission.csv`.
