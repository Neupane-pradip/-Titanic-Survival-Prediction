<<<<<<< HEAD
# -Titanic-Survival-Prediction
=======
# Titanic Survival Prediction Project

This project is a small machine learning workflow built on the Titanic dataset.
It teaches the full path from exploring data, to training models, to reading
results in a simple way.

## What this project does

- loads the Titanic training data from `data/raw/train.csv`
- cleans and encodes the data in `src/preprocess.py`
- trains and compares `LogisticRegression`, `RandomForest`, and
  `GradientBoosting`
- evaluates the models with accuracy, precision, recall, F1, and confusion matrices
- compares baseline features vs engineered features
- saves charts, logs, and experiment history in `outputs/`

## Setup

If you are using the existing virtual environment, activate it first.
Then install the dependencies:

```powershell
pip install -r requirements.txt
```

## Run the project

### Train and compare models

```powershell
python main.py
```

This will:
- run a train/test split
- compare baseline vs engineered features
- print model metrics and subgroup error summaries
- save confusion matrices, feature importance files, comparison CSVs, and the
  experiment log

### Create the simple visualization dashboard

```powershell
python visualize.py
```

This saves the overview chart to `outputs/titanic_overview.png`.

## Notebooks

Use the notebooks in this order if you want to learn the project step by step:

1. `notebooks/01_eda.ipynb` — basic exploration and a few simple plots
2. `notebooks/02_experiment_log_summary.ipynb` — a short readable summary of the experiment history
3. `notebooks/03_experiment_log_dashboard.ipynb` — an interactive log dashboard with widgets

If notebook widgets are needed, install the notebook extras:

```powershell
pip install -r requirements.txt
```

## Outputs

Running `main.py` creates or updates these artifacts:

- `outputs/confusion_matrices/<scenario>/`
- `outputs/feature_importance/<scenario>/`
- `outputs/subgroup_errors/<scenario>/`
- `outputs/feature_engineering_comparison.csv`
- `outputs/experiment_log.csv`
- `outputs/experiment_history.png`

Running `visualize.py` creates:

- `outputs/titanic_overview.png`

## Project flow

The code is organized so you can follow the workflow easily:

- `src/preprocess.py` prepares the Titanic data and adds engineered features
- `src/train.py` defines the models used in the comparison
- `src/evaluate.py` computes metrics and saves evaluation plots
- `main.py` runs the full experiment loop and logs results
- `visualize.py` creates a simple overview dashboard

## A simple learning path

If you are studying the project, follow this order:

1. Open `notebooks/01_eda.ipynb` and understand the data
2. Run `python main.py` and review the printed metrics
3. Open `notebooks/02_experiment_log_summary.ipynb` to see the overall trend
4. Open `notebooks/03_experiment_log_dashboard.ipynb` for interactive filtering
5. Compare the baseline and engineered feature results in `outputs/feature_engineering_comparison.csv`

That sequence gives you a complete beginner-friendly view of the project from
data exploration to model comparison and experiment tracking.

>>>>>>> 531c88a (Initial full Titanic project)
